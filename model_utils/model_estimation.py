 # ds libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,r2_score,mean_squared_error,median_absolute_error,max_error
from sklearn.metrics import mean_gamma_deviance,mean_poisson_deviance,explained_variance_score,mean_squared_log_error
from sklearn.metrics import roc_auc_score,accuracy_score,fbeta_score,precision_score,recall_score,confusion_matrix,average_precision_score
from sklearn.preprocessing import KBinsDiscretizer

# вспомогательный функции
########################################################################################################################################################################################################
def find_table_for_feature(feature,table2feature):
    for t,f in table2feature.items():
        if feature in f:
            return t
        
def discretize_numeric_column(train_column: pd.Series,column=None,discretizer=None):
    '''Функция обучения дискертизатора и дискретизации числовой колонки
    ----------------------------------------------------------------------------------
    Parametrs:
    train_column: np.array shape (n,1)
        вектор, на чем обучать дискертизатор
    column: pd.Series/np.array
        вектор, который будет подвергнут дискретизации
    discretizer: KBinsDiscretizer
        KBinsDiscretizer настроенный по собсвтенному желанию или None чтобы по умолчанию
    -------------------------------------------------------------------------------------
    Return: pd.Series, KBinsDiscretizer
    '''
    if not isinstance(discretizer,KBinsDiscretizer):
        discretizer =  KBinsDiscretizer(n_bins=4,strategy='quantile',encode = 'ordinal',dtype=np.float32)        
    # обучение бинарайзера
    discretizer.fit(train_column)
    if column is None:
        column = train_column
    # получение лейблов с рабивкой по бинам на тесте
    bin_column = discretizer.transform(column)
 
    return bin_column, discretizer

# функции расчет метрик под задачу
########################################################################################################################################################################################################
def reg_model_metric_perfomance(y_true, y_pred, **kwargs):
    '''
    Функция, которая оценивает регрессионную модель по основным метрикам
    ----------------------------------------------------------------------
    Parametrs:
    y_true: np.array/pd.Series
        вектор целевой переменной
    y_pred: np.array/pd.Series
        вектор предсказаний
    round_n: int
        до какого порядка округляем
    -------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    round_n = kwargs.get('round_n', None)
    # округлим до n порядка
    if isinstance(round_n,int):
        y_pred = round_n * np.round(y_pred / round_n)
    # расчет основных метрик
    reg_metrics = {
            'N': len(y_pred),
            'std_target': y_true.std(), 'mean_target': y_true.mean(), f'median_target': np.median(y_true),
            'std_pred': y_pred.std(), f'mean_pred': y_pred.mean(), 'median_pred': np.median(y_pred),
            'medianae': median_absolute_error(y_true, y_pred), 'maxae': max_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred), 'mae': mean_absolute_error(y_true, y_pred),
            'rmse': mean_squared_error(y_true, y_pred, squared=False), 'rmsle': mean_squared_log_error(y_true, y_pred, squared=False),
            'R2': r2_score(y_true, y_pred), 'explained_variance': explained_variance_score(y_true, y_pred),
            'gamma': mean_gamma_deviance(y_true,y_pred), 'poisson': mean_poisson_deviance(y_true, y_pred),
            'medianae / median': median_absolute_error(y_true, y_pred) / np.median(y_true), 
            'mae / mean': mean_absolute_error(y_true, y_pred) / np.mean(y_true)
        }
    
    return reg_metrics

def binary_model_metric_perfomance(y_true, y_pred,**kwargs):
    '''
    Функция, которая оценивает модель multiclass по основным метрикам
    ----------------------------------------------------------------------
    Parametrs:
    y_true: np.array/pd.Series
        вектор целевой переменной
    y_pred: np.array/pd.Series
        вектор предсказаний
    threshold: int
        порог для бинаризаци предсказний модели
    -------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    threshold = kwargs.get('threshold', 0.5)
    
    pred = (y_pred >= threshold).astype(int)
    
    target_distrib = np.unique(y_true,return_counts=True)
    pred_distrib = np.unique(pred,return_counts=True)
    
    target_classes = {f"target_label_count_{target_distrib[0][i]}" : target_distrib[1][i] for i in range(len(target_distrib[0]))}
    pred_classes = {f"pred_label_count_{pred_distrib[0][i]}" : pred_distrib[1][i] for i in range(len(pred_distrib[0]))}
    
    binary_metrics = {
            'N': len(pred),
            'accuracy' : accuracy_score(y_true, pred),
            'precision' : precision_score(y_true, pred),
            'recall' : recall_score(y_true, pred),
            'f1' : fbeta_score(y_true, pred, beta=1),
            'roc-auc original':roc_auc_score(y_true, y_pred),
            'roc-auc_hack':roc_auc_score(y_true, pred),
            'pr-auc':average_precision_score(y_true, y_pred),
        }
    
    binary_metrics.update(target_classes)
    binary_metrics.update(pred_classes)
        
    return binary_metrics

def multiclass_model_metric_perfomance(y_true, y_pred,**kwargs):
    '''
    Функция, которая оценивает модель multiclass по основным метрикам
    ----------------------------------------------------------------------
    Parametrs:
    y_true: np.array/pd.Series
        вектор целевой переменной
    y_pred: np.array/pd.Series
        вектор предсказаний
    -------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    pred = np.argmax(pred_proba,axis = 1)
    
    target_distrib = np.unique(y_true,return_counts=True)
    pred_distrib = np.unique(pred,return_counts=True)
    
    target_classes = {f"target_label_count_{target_distrib[0][i]}" : target_distrib[1][i] for i in range(len(target_distrib[0]))}
    pred_classes = {f"pred_label_count_{pred_distrib[0][i]}" : pred_distrib[1][i] for i in range(len(pred_distrib[0]))}
    
    multiclass_metrics = {
            'N': len(pred),
            'accuracy' : accuracy_score(y_true,pred),
            'precision_micro' : precision_score(y_true,pred,average='micro'), 'precision_macro' : precision_score(y_true,pred,average='macro'),
            'recall_micro' : recall_score(y_true,pred,average='micro'), 'recall_macro' : recall_score(y_true,pred,average='macro'),
            'f1_micro' : fbeta_score(y_true,pred,average='micro',beta=1), 'f1_macro' : fbeta_score(y_true,pred,average='macro',beta=1)
        }
    
    multiclass_metrics.update(target_classes)
    multiclass_metrics.update(pred_classes)
        
    return multiclass_metrics

# итоговые функции подсчета метрик под модель
########################################################################################################################################################################################################
def model_metric_perfomance(automl, dataset,cv_iter=None, **kwargs):
    '''
    Функция, которая считает метрики для automl
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Parametrs:
    automl: lightautoml
        основная обученная модель
    dataset: pd.DataFrame
        датасет с bin_col, по этой переменной будем разбиение на группы и считать в ней метрики
    cv_iter: cv_iter from sklearn.model_selection. KFOLD, StratifiedKFold
        по нему будем разбивать на фолды и замерять метрики на тестовой части, чтобы проверить стабильность метрик
    kwargs: round_n, threshold
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    
    task_metrics_func = {'reg' : reg_model_metric_perfomance, 'binary' : binary_model_metric_perfomance, 'multiclass' : multiclass_model_metric_perfomance}
    
    task_name = automl.task.name
    target_name = automl.reader.target
    
    # получим предсказания и целевые переменные для последующего использования в метриках
    if task_name == 'reg':
        automl_predict = automl.predict(dataset).data.reshape(-1)
        target = dataset[target_name].values
    elif task_name == 'binary':
        automl_predict = automl.predict(dataset).data.reshape(-1)
        target = dataset[target_name]#automl.reader.check_class_target(dataset[target_name])[0].values
    elif task_name == 'multiclass':
        automl_predict = automl.predict(dataset).data
        target = automl.reader.check_class_target(dataset[target_name])[0].values
        
    # расчет метрик
    metrics = task_metrics_func[task_name](target, automl_predict, **kwargs)
    # расчет целевой метрики из обучения
    metrics['task_' + 'func' if automl.task.metric_name is None else automl.task.metric_name] = automl.task.metric_func(target,automl_predict)
    if cv_iter is not None:
        cv_metrics = []
        for i, (train_index, test_index) in enumerate(cv_iter.split(target)):
            # возьмем конкретный фолд 
            target_fold = target[test_index]
            pred_fold = automl_predict[test_index]
            # расчет метрик
            fold_metrics = task_metrics_func[task_name](target_fold, pred_fold, **kwargs)
            # расчет целевой метрики из обучения
            fold_metrics['task_'+automl.task.metric_name] = automl.task.metric_func(target_fold,pred_fold)
            fold_metrics['fold'] = i + 1
            
            cv_metrics.append(fold_metrics)
            
        # добавим расчитанные метрики по фолдам        
        cv_metrics = pd.DataFrame(cv_metrics)
        metrics['fold_metrics'] = cv_metrics
        drop_columns = ['N','fold'] + [el for el in metrics.keys() if 'label_count' in el] 
        metrics['mean_metrics'] = cv_metrics.drop(columns=drop_columns).mean()
        metrics['std_metrics'] = cv_metrics.drop(columns=drop_columns).std()
        
    return metrics
    
def model_metric_perfomance_on_bins(automl, dataset, bin_col,**kwargs):
    '''
    Функция, которая считает метрики для каждого значения в bin_col
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Parametrs:
    automl: lightautoml
        основная обученная модель
    dataset: pd.DataFrame
        датасет с bin_col, по этой переменной будем разбиение на группы и считать в ней метрики
    bin_col: str
        название переменной, по которой определяем разбиение на бины
        
    kwargs: discretizer,round_n, threshold
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Return:
        dict with metrics
    '''
    edges = None
    if isinstance(kwargs.get('discretizer',None),KBinsDiscretizer):
        discretizer, round_n = kwargs['discretizer'], kwargs.get('round_n',1)
        # получение пороговых значений c округлением 
        edges = round_n * np.round(discretizer.bin_edges_[0] / round_n) 
    #     хранение результатов
    metric_results = {}
    # цикл подсчета метрик по бину
    for bin_ in np.sort(dataset[bin_col].unique()):
        bin_dataset = dataset[dataset[bin_col] == bin_] 
        if len(bin_dataset):
            name = f"{bin_col} with bin = {bin_}"
            if edges is not None:
                name += f' from {edges[int(bin_)]:_} to {edges[int(bin_+1)]:_}'
            metric_results[name] = model_metric_perfomance(automl,bin_dataset,**kwargs)
        
    return metric_results 

# оценка значимости признаков
######################################################################################################################################################################################################   
def estimate_model_feature_importance(model,test_data=None,holdout_data=None,savefile_excel=None,silent=True):
    '''
    Функция, которая замеряет
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Parametrs:
    model: lightautoml
        основная обученная модель
    test_data: pd.DataFrame
        датасет с целевой переменной на котором считаем метрики но в отложенном множестве но отдельном теством множестве, но примеры взяты из обучающего временного промежутка
    holdout_data: pd.DataFrame
        датасет с целевой переменной на котором считаем метрики но в отложенном множестве
    savefile_excel: str
        название файла для сохранения фичей в формате xlsx
    silent: bool 
        Выводить процесс рассчета значимости фичей
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Return:
        pd.DataFrame со оцененными фичами
    '''
    # lama automl
    if 'report_deco' in model.__module__:
        model = model.model
    
    if isinstance(model.get_feature_scores('fast'),pd.DataFrame):
        model_feature_importance = model.get_feature_scores('fast').rename(columns={'Importance':'Train Importance'})
    else:
        model_feature_importance = pd.DataFrame(data = model.reader.used_features,columns=['Feature'])
        
    # оценка на тестовом множестве значимых фичей
    if test_data is not None:
        if model.task.name == 'multiclass':
            test_data[model.reader.target] = model.reader.check_class_target(test_data[model.reader.target])[0].values
            
        test_feature_importance_dict = model.get_feature_scores('accurate',test_data,silent=silent).set_index('Feature')['Importance'].to_dict()
        model_feature_importance['Val Importance'] = model_feature_importance.Feature.map(lambda x: test_feature_importance_dict.get(x,None))
    # оценка на отложенном множестве значимых фичей   
    if holdout_data is not None:
        if model.task.name == 'multiclass':
            holdout_data[model.reader.target] = model.reader.check_class_target(holdout_data[model.reader.target])[0].values
        holdout_feature_importance_dict = model.get_feature_scores('accurate',holdout_data,silent=silent).set_index('Feature')['Importance'].to_dict()
        model_feature_importance['Holdout Importance'] = model_feature_importance.Feature.map(lambda x: holdout_feature_importance_dict.get(x,None))
    # сохранение в эксель
    if savefile_excel is not None and str(savefile_excel).split('.')[-1] == 'xlsx':
            model_feature_importance.to_excel(savefile_excel,index=False)
        
    return model_feature_importance

def compute_shap_values(lama_model,data):
    '''Функция расчета shap values для каждого типа бустинговых моделей 1-level automl'''
    
    reader = lama_model.reader
    shaps = {}
    
    for level in lama_model.levels[0]:
        if 'RFSklearn' in level.ml_algos[0].name:
            continue
            
        features_pipeline = level.features_pipeline
        df = features_pipeline.transform(reader.read(data)).to_pandas().data
        
        for algo in level.ml_algos:
            shaps[algo.name] = {}
            for i,algo_model in enumerate(algo.models):
            
                explainer = shap.TreeExplainer(algo_model)

                if 'CatBoost' in algo.name:
                    shap_values = explainer(df,data[reader.target])
                    shaps[algo.name][i] = (shap_values,df)

                elif 'LightGBM' in algo.name or 'RFSklearn' in algo.name:
                    shap_values = explainer.shap_values(df)
                    shaps[algo.name][i] = (shap_values,df)
            
    return shaps
    





