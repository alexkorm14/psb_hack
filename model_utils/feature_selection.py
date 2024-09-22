import operator
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import median_absolute_error
import xgboost as xgb
from lightautoml.tasks import Task
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures

################################################################################################################################################################################
# prepare data for feature selection

def prepare_datasets_for_feature_selection(train, val, config_model):
    '''Функция, которая подготовит train, val для последующего отбора фичей'''
    # через Reader сделаем датасет для обработки фичей через LGBSimpleFeatures
    # create Task
    task = Task(name=config_model['task']['name'], 
                metric=config_model['task']['metric'] if config_model['task']['metric'] != 'medianae' else median_absolute_error, 
                loss=config_model['task']['loss'], 
                greater_is_better=config_model['task']['greater_is_better'])
    # create PandasToPandasReader
    reader_params = {'n_jobs' : config_model['n_threads'], 'cv': config_model['n_folds'], 'random_state' : config_model['random_state']}
    reader = PandasToPandasReader(task = task, advanced_roles=False, **reader_params)
    # обучим reader
    X_train = reader.fit_read(train, roles = config_model['roles'])
    # preprocess for feature selection
    feature_preprocess = LGBSimpleFeatures()
    X_train = feature_preprocess.fit_transform(X_train).to_pandas().data
    # сделаем категориями категориальные фичи
    cat_cols = list(filter(lambda x: 'ord__' in x, X_train.columns))
    X_train = X_train.astype(dict(zip(cat_cols,['str'] * len(cat_cols))))
    X_train.columns = list(map(lambda x: x.split('ord__')[-1],X_train.columns))
    y_train = train[config_model['roles']['target']]
    
    X_val = reader.read(val)
    X_val = feature_preprocess.transform(X_val).to_pandas().data
    # сделаем категориями категориальные фичи
    cat_cols = list(filter(lambda x: 'ord__' in x, X_val.columns))
    X_val = X_val.astype(dict(zip(cat_cols,['str'] * len(cat_cols))))
    X_val.columns = list(map(lambda x: x.split('ord__')[-1],X_val.columns))
    y_val = val[config_model['roles']['target']]
    
    return X_train, y_train, X_val, y_val


################################################################################################################################################################################
# BoostARoota Main Class and Methods
class BoostARoota(object):

    def __init__(self, metric=None, cutoff=4, iters=10,
                 max_rounds=100, delta=0.1, random_state=None,
                 device='cpu'):
        self.metric = metric
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.random_state = random_state
        self.device = device
        self.keep_vars_ = None

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if metric is None:
            raise ValueError('you must enter metric')
        if cutoff <= 0:
            raise ValueError('cutoff should be greater than 0. You entered' + str(cutoff))
        if iters <= 0:
            raise ValueError('iters should be greater than 0. You entered' + str(iters))
        if (delta <= 0) or (delta > 1):
            raise ValueError('delta should be between 0 and 1, was ' + str(delta))

        # Issue warnings for parameters to still let it run
        if delta < 0.02:
            warnings.warn("WARNING: Setting a delta below 0.02 may not converge on a solution.")
        if max_rounds < 1:
            warnings.warn("WARNING: Setting max_rounds below 1 will automatically be set to 1.")

    def fit(self, x, y):
        self.keep_vars_ = _BoostARoota(x, y,
                                       metric=self.metric,
                                       cutoff=self.cutoff,
                                       iters=self.iters,
                                       max_rounds=self.max_rounds,
                                       delta=self.delta,
                                       device=self.device,
                                       random_state=self.random_state)
        np.random.seed(None)

        return self

    def transform(self, x):
        if self.keep_vars_ is None:
            raise ValueError("You need to fit the model first")
        return x[self.keep_vars_]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


################################################################################################################################################################################
# Helper Functions to do the Heavy Lifting
def _create_shadow(x_train, random_state):
    """
    Take all X variables, creating copies and randomly shuffling them
    :param x_train: the dataframe to create shadow features on
    :return: dataframe 2x width and the names of the shadows for removing later
    """
    if random_state is not None:
        np.random.seed(random_state)

    x_shadow = x_train.copy()
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)
    # rename the shadow
    shadow_names = ["ShadowVar" + str(i + 1) for i in range(x_train.shape[1])]
    x_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([x_train, x_shadow], axis=1)

    return new_x, shadow_names

def _reduce_vars_xgb(x, y, metric, this_round, cutoff, n_iterations, delta, device, random_state):
    """
    Function to run through each
    :param x: Input dataframe - X
    :param y: Target variable
    :param metric: Metric to optimize in XGBoost
    :param this_round: Round so it can be printed to screen
    :return: tuple - stopping criteria and the variables to keep
    """
    # Set up the parameters for running the model in XGBoost - split is on multi log loss
    if metric == 'mlogloss':
        param = {'objective': 'multi:softmax',
                 'eval_metric': 'mlogloss',
                 'num_class': len(np.unique(y)),
                 'device': device}
    else:
        param = {'eval_metric': metric,
                 'device': device,}

    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances

        new_x, shadow_names = _create_shadow(x,
                                             random_state=((random_state + i) if random_state is not None else None))
        dtrain = xgb.DMatrix(new_x, label=y,enable_categorical=True)
        bst = xgb.train(param, dtrain, verbose_eval=False)
        if i == 1:
            df = pd.DataFrame({'feature': new_x.columns})
            pass

        importance = bst.get_score(importance_type='gain')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df2 = pd.DataFrame(importance, columns=['feature', 'fscore' + str(i)])
        df2['fscore' + str(i)] = df2['fscore' + str(i)] / df2['fscore' + str(i)].sum()
        df = pd.merge(df, df2, on='feature', how='outer')
        
        print("Round: ", this_round, " iteration: ", i)

    df['Mean'] = df[df.columns[1:]].rank(0).mean(1)

    # Split them back out
    real_vars = df[~df['feature'].isin(shadow_names)]
    shadow_vars = df[df['feature'].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars['Mean'].mean() / cutoff
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]

    # Check for the stopping criteria
    # Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if (len(real_vars['feature']) / len(x.columns)) > (1 - delta):
        criteria = True
    else:
        criteria = False

    return criteria, real_vars['feature']

# Main function exposed to run the algorithm
def _BoostARoota(x, y, metric, cutoff, iters, max_rounds, delta, device, random_state):
    """
    Function loops through, waiting for the stopping criteria to change
    :param x: X dataframe One Hot Encoded
    :param y: Labels for the target variable
    :param metric: The metric to optimize in XGBoost
    :return: names of the variables to keep
    """

    new_x = x.copy()
    # Run through loop until "crit" changes
    i = 0
    while True:
        # Inside this loop we reduce the dataset on each iteration exiting with keep_vars
        i += 1
        crit, keep_vars = _reduce_vars_xgb(new_x,
                                           y,
                                           metric=metric,
                                           this_round=i,
                                           cutoff=cutoff,
                                           n_iterations=iters,
                                           delta=delta,
                                           device=device,
                                           random_state=random_state)

        if crit | (i >= max_rounds):
            break  # exit and use keep_vars as final variables
        else:
            new_x = new_x[keep_vars].copy()
    
    print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
    return keep_vars
