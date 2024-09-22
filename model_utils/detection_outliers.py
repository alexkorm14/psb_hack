from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
import numpy as np

class IqrDetector(BaseEstimator):
    def __init__(self,outlier_column):
        self.outlier_column = outlier_column
        
    def fit(self,X,y=None):
        data = X[self.outlier_column]
        self.q1,self.q3 = data.quantile(0.25), data.quantile(0.75)
        self.iqr = self.q3-self.q1
        
        return self
    
    def predict(self,X,y=None):
        data = X[self.outlier_column]
        
        upper_bound = self.q3 + 1.5 * self.iqr
        lower_bound = self.q1 - 1.5 * self.iqr
        
        predict = ((data >= upper_bound) | (data <= lower_bound)).astype(int)
        
        return predict
        
class SigmaDetector(BaseEstimator):
    def __init__(self,outlier_column,z_threshold):
        self.outlier_column = outlier_column
        self.z_threshold = z_threshold
        
    def fit(self,X,y=None):
        data = X[self.outlier_column]
        self.mean,self.std = data.mean(), data.std()
     
        return self
    
    def predict(self,X,y=None):
        data = X[self.outlier_column]
        
        z_score = ((data - self.mean) / self.std).abs()
        
        predict = (z_score > self.z_threshold).astype(int)
        
        return predict

class OutlierRemover(BaseEstimator,TransformerMixin):
    def __init__(self,outlier_column,method,**kwargs):
        self.outlier_column = outlier_column
        self.method = method
        self.kwargs = kwargs
        self.anomaly_model = None
        
    def fit(self,X,y=None):
        if self.method == 'iqr':
            self.anomaly_model = IqrDetector(self.outlier_column)
            self.anomaly_model.fit(X)
        elif self.method == 'z-score':
            z_threshold = self.kwargs.get('z_threshold',2)
            self.anomaly_model = SigmaDetector(self.outlier_column,z_threshold)
            self.anomaly_model.fit(X)
            
        elif self.method in ['isolation_forest','lof']:
            method_algo = {'isolation_forest' : IsolationForest,'lof' : LocalOutlierFactor}
            if self.kwargs.get('additional_columns',None):
                self.additional_columns = self.kwargs['additional_columns']
                self.kwargs.pop('additional_columns')
                data = X[self.additional_columns + [self.outlier_column]]
            else:
                data = data.values.reshape(-1,1)

            self.anomaly_model = method_algo[self.method](**self.kwargs)
        
            self.anomaly_model.fit(data)
        
        return self
    
    def transform(self,X,y=None):
        if self.method in ('iqr','z-score'):
            outliers = self.anomaly_model.predict(X)
            return X.loc[outliers == 0]
        elif self.method in ['isolation_forest','lof']:
            outliers = self.anomaly_model.predict(X[self.additional_columns + [self.outlier_column]])
            return X.loc[outliers > 0]
        

def find_outliers(df,outlier_columns,method='iqr',**kwargs):
    '''
    Функция которая возвращает индексы элементов назначенного поля, которые определяются выбранным методом как выбросы
    -------------------------------------------------------------------------------------------------------------------------------
    Parametrs:
     - df: pd.DataFrame 
         основной датасет, в которо надо обнаружить выбросы
     - outlier_column: str 
         поле в котором надо искать выбросы
     - method: str
         метод, которомы мы хотим определить выбросы. Доступные методы: 'iqr', 'z-score', 'isolation_forest', 'lof',
    -----------------------------------------------------------------------------------------------------------------------------------
    Return: np.array
        индексы элемтнов, которые были распознаны как выбросы
    '''
    data = df[outlier_columns]
    
    if method == 'iqr':
        q1,q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3-q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        idx = np.where((data > upper_bound)|(data < lower_bound)) 
        
    elif method == 'z-score':
        z_threshold = kwargs.get('z_threshold',2)
        mean,std = data.mean(), data.std()
        z_score = ((data - mean) / std).abs()
        idx = np.where(z_score > z_threshold)
        
    elif method in ['isolation_forest','lof']:
        method_algo = {'isolation_forest' : IsolationForest,'lof' : LocalOutlierFactor}
        if kwargs.get('additional_columns',None):
            additional_columns = kwargs['additional_columns']
            data = df[[outlier_columns] + additional_columns]
            kwargs.pop('additional_columns')
        else:
            data = data.values.reshape(-1,1)

        outliers_algo = method_algo[method](**kwargs)
        outliers = outliers_algo.fit_predict(data)
        idx = np.where(outliers == -1)
    
    return idx[0]              

def find_high_nan_constant_features(df, max_nan_rate=0.95, max_constant_rate=0.95, cv_iter=None):
    '''Функция выявления признаков с высокой долей пропусков или одного константного значения. Возвращает список таких признаков'''
    if cv_iter is None:
        # если не передан cv_iter то посчитаем долю пропусков и долю популярного констатного значения для каждого признака на всем датасете 
        const_nan_rates = pd.DataFrame({
            'columns' : df.columns,
            'constant_rate' : [df[x].isin(df[x].mode()).mean() for x in df.columns],
            'nan_rate' : [df[x].isna().mean() for x in df.columns]
        })
        # определим по заданным критериям признаки 
        cond = (const_nan_rates['constant_rate'] >= max_constant_rate) | (const_nan_rates['nan_rate'] >= max_nan_rate)
        drop_features = const_nan_rates.loc[cond,'columns'].values.tolist()
    else:
        # передан cv_iter, значит на каждом фолде посчитаем долю пропусков и долю популярного констатного значения для каждого признака и потом найдем пересечение 
        # cv_iter передается для большей надежности определения таких признаков
        drop_features = set(df.columns.values.tolist())
        for i, (train_index, test_index) in enumerate(cv_iter.split(df)):
            # возьмем конкретный фолд 
            fold = df.iloc[train_index]
            # посчитаем долю пропусков и долю популярного констатного значения для каждого признака на этом фолде
            const_nan_rates = pd.DataFrame({
            'columns' : fold.columns,
            'constant_rate' : [fold[x].isin(fold[x].mode()).mean() for x in fold.columns],
            'nan_rate' : [fold[x].isna().mean() for x in fold.columns]
            })
            # определим по заданным критериям признаки на фолде
            cond = (const_nan_rates['constant_rate'] >= max_constant_rate) | (const_nan_rates['nan_rate'] >= max_nan_rate)
            # пересечем с признаками, которые надо выкинуть
            drop_features &= set(const_nan_rates.loc[cond,'columns'].values.tolist())
        
    return sorted(drop_features)