import pandas as pd
import numpy as np
import os
import sys
import joblib
import re
from tqdm import tqdm

#######################################################################################################################################################################################
# признаки агрегаты во времени
def calc_mean_target_features(df, output_file_name, group_columns, window_sizes, date_col_name, min_objects_in_window=1, target_col_name="target"):
    """
    Окно для каждой даты-времени смотрит назад на window_size дней
    Для каждой даты мы возьмем последнее значение и сохраним
    Дата выкидывается, если в ее окне оказалось меньше min_objects_in_window наблюдений
    :param group_columns: колонки, по которым сгруппировать
    :param target_col_name: имя колонки таргета
    :param date_col_name: имя колонки даты
    :param output_file_name: имя выходного файла
    
    Статистики мапятся в выходной файл словарем в формате
    Группировочная колонка: статистика: размер окна (например, 5D): дата: значение
    value = dict[col][stat][window][str(date)]
    """
    df = df.copy()
    split_string = "-"

    df[date_col_name] = pd.to_datetime(df[date_col_name])
    df = df.set_index(date_col_name).sort_index()
    added_features_names = []
    
    for col_name in group_columns:
        for window_size in window_sizes:
            feature_name = split_string + f"{window_size}D" + split_string + col_name
            
            avg_feature_name = "avg" + feature_name
            added_features_names.append(avg_feature_name)
            df[avg_feature_name] = df[target_col_name].rolling(window=f"{window_size}D", min_periods=min_objects_in_window).mean()
            
            var_feature_name = "var" + feature_name
            added_features_names.append(var_feature_name)
            df[var_feature_name] = df[target_col_name].rolling(window=f"{window_size}D", min_periods=min_objects_in_window).std()
    
    df = df.reset_index(drop=False).drop_duplicates(subset=[date_col_name], keep='last').set_index(date_col_name)
    calculated_stats = {}
    
    for feature_name in added_features_names:
        stat, window, col_name = feature_name.split(split_string)
        for date in tqdm(df.index, desc=f"processing {col_name}_{window}_{stat}"):
            if col_name not in calculated_stats:
                calculated_stats[col_name] = {}
            if stat not in calculated_stats[col_name]:
                calculated_stats[col_name][stat] = {}
            if window not in calculated_stats[col_name][stat]:
                calculated_stats[col_name][stat][window] = {}
            calculated_stats[col_name][stat][window][date] = df.loc[date, feature_name]
            
    with open(output_file_name, "wb") as fp:
        joblib.dump(calculated_stats , fp)

def add_mean_target_features(df, file_name, group_columns, window_sizes, date_col_name, target_col_name="target"):
    #     value = dict[col][stat][window][str(date)]
    df = df.copy()
    # df[date_col_name] = df[date_col_name].astype("str")
    with open(file_name, "rb") as json_file:
        stats = joblib.load(json_file)
        for col in group_columns:
            if col in stats:
                for window_size in window_sizes:
                    window_size = f"{window_size}D"
                    if window_size in stats[col]["avg"]:
                        col_name = f"avg_{target_col_name}_by_" + col + "_" + window_size
                        dates_avg = pd.Series(stats[col]["avg"][window_size])
                        dates_avg.name = col_name
                        df = pd.merge(df, dates_avg, left_on=date_col_name, right_index=True, how="left")
                        
                    if window_size in stats[col]["var"]:
                        col_name = f"var_{target_col_name}_by_" + col + "_" + window_size
                        dates_var = pd.Series(stats[col]["var"][window_size])
                        dates_var.name = col_name
                        df = pd.merge(df, dates_var, left_on=date_col_name, right_index=True, how="left")        
        return df


#######################################################################################################################################################################################
# признаки по времени
def date_features(df):
    '''
    Признаки сгенерированные из временных полей:
    - Дата бронирования
    - Заезд
    - Выезд
    Суть признаков: 
    - посчитать месяц, квартал, сезон в этих датах
    - выявить разницу в днях между бронированием и выявить флаги бизнесовые на основе разниц между датами
    '''
    # вспомогательные словари
    datecol2eng_map = {'Дата бронирования':'booking','Заезд':'checkin','Выезд':'checkout'}
    season_map_dict = {1 : 'winter', 2 : 'winter', 12: 'winter',
                        3: 'spring', 4: 'spring', 5: 'spring',
                        6: 'summer', 7: 'summer', 8: 'summer',
                        9: 'autumn', 10: 'autumn',11: 'autumn'}                       
    # Конвертация дат в формат datetime
    df['Дата бронирования'] = pd.to_datetime(df['Дата бронирования'])
    df['Заезд'] = pd.to_datetime(df['Заезд'])
    df['Выезд'] = pd.to_datetime(df['Выезд'])
    # посчитать месяц, квартал, сезон в этих датах
    for date_col in ['Дата бронирования','Заезд','Выезд']:
        # Создание новых признаков
        eng_name = datecol2eng_map[date_col]
        df[f'{eng_name}_year'] = df[date_col].dt.year
        df[f'{eng_name}_month'] = df[date_col].dt.month
        df[f'{eng_name}_day'] = df[date_col].dt.day
        df[f'{eng_name}_dayofweek'] = df[date_col].dt.dayofweek
        df[f'{eng_name}_weekofyear'] = df[date_col].dt.isocalendar().week
        df[f'{eng_name}_is_weekend'] = df[f'{eng_name}_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df[f'{eng_name}_quarter'] = df[date_col].dt.quarter
        df[f'{eng_name}_quarter'] = df[date_col].dt.quarter
        df[f'{eng_name}_season'] = df[f'{eng_name}_month'].map(lambda x: season_map_dict[x])
        df[f'{eng_name}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df[f'{eng_name}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df[f'{eng_name}_is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df[f'{eng_name}_is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        df[f'{eng_name}_is_year_start'] = df[date_col].dt.is_year_start.astype(int)
        df[f'{eng_name}_is_year_end'] = df[date_col].dt.is_year_end.astype(int)
        df[f'{eng_name}_hour'] = df[date_col].dt.hour
        # Синус и косинус дня недели (для учета цикличности)
        df[f'{eng_name}_dayofweek_sin'] = np.sin(2 * np.pi * df[f'{eng_name}_dayofweek'] / 7)
        df[f'{eng_name}_dayofweek_cos'] = np.cos(2 * np.pi * df[f'{eng_name}_dayofweek'] / 7)
        
        # Синус и косинус месяца
        df[f'{eng_name}_month_sin'] = np.sin(2 * np.pi * df[f'{eng_name}_month'] / 12)
        df[f'{eng_name}_month_cos'] = np.cos(2 * np.pi * df[f'{eng_name}_month'] / 12)
        # sin cos hour
        df[f'{eng_name}_hour_sin'] = np.sin(2 * np.pi * df[f'{eng_name}_hour'] / 24)
        df[f'{eng_name}_hour_cos'] = np.cos(2 * np.pi * df[f'{eng_name}_hour'] / 24)

    # разница между датами
    df['booking_to_checkin_days'] = (df['Заезд'] - df['Дата бронирования']).dt.days
    df['booking_to_checkin_months'] = df['booking_to_checkin_days'] // 30
    df['checkin_to_checkout_days'] = (df['Выезд'] - df['Заезд']).dt.days

    # Флаги раннего/позднего бронирования
    df['is_last_week_booking'] = df['booking_to_checkin_days'].apply(lambda x: 1 if x <= 7 else 0)
    df['is_early_booking'] = df['booking_to_checkin_days'].apply(lambda x: 1 if x >= 30 else 0)

    df['Дата бронирования'] = pd.to_datetime(df['Дата бронирования'].dt.date)
    df['Заезд'] = pd.to_datetime(df['Заезд'].dt.date)
    df['Выезд'] = pd.to_datetime(df['Выезд'].dt.date)

    drop_columns = ['booking_to_checkin_days']

    return df.drop(columns=drop_columns)

#######################################################################################################################################################################################
# признаки по категории номера
def category_room_features(df):
    '''
    Признаки сгенерированные из 'Категория номера':
    - можно вытащить количество типов номеров в заказе: сколько было забронировано Студий, Стандартов, Люксов, Коттеджей
    - сколько спален в броне всего
    - флаг номера для малоактивных гостей
    - флаг отдельного входа
    - флаг брони разных номеров
    '''
    splitter = re.compile("([0-9]+)\. ")
      # Создание признаков из 'Категория номера'
    df['Категория номера'] = df['Категория номера'].astype(str)
    # распарсить признак Категория номера
    df['rooms'] = df['Категория номера'].map(lambda x: list(filter(lambda y: not y.isdigit() and y != '', re.split(splitter,x.replace('\n',' ')))))
    df['rooms'] = df['rooms'].map(lambda x: list(map(str.strip,x)))
    # количество типов номеров в заказе: сколько было забронировано Студий, Стандартов, Люксов, Коттеджей
    df['has_apart'] = df['rooms'].map(lambda x: sum(map(lambda y: 'Апарт' in y,x)))
    df['has_luxe'] = df['rooms'].map(lambda x: sum(map(lambda y: 'Люкс' in y,x)))
    df['has_standard'] = df['rooms'].map(lambda x: sum(map(lambda y: 'Стандарт' in y,x)))
    df['has_studio'] = df['rooms'].map(lambda x: sum(map(lambda y: 'Студия' in y,x)))
    df['has_cottage'] = df['rooms'].map(lambda x: sum(map(lambda y: 'Коттедж' in y,x)))
    # сколько спален
    df['count_bedrooms'] = df['rooms'].map(lambda x: sum(map(lambda y: int(re.findall(r'\d+',y)[0]) if re.findall(r'\d+',y) else 1,x)))
    # флаги
    df['has_low_active'] = df['rooms'].map(lambda x: int(sum(map(lambda y: 'маломобильных' in y,x)) > 0))
    df['has_own_entarance'] = df['rooms'].map(lambda x: int(sum(map(lambda y: 'отдельным входом' in y,x)) > 0))
    df['is_all_rooms_same'] = df['rooms'].map(lambda x: int(len(set(x)) == 1))
  
    drop_columns = ['rooms']

    return df.drop(columns=drop_columns)

#######################################################################################################################################################################################
# признаки на основе стоимости и количества гостей

def cost_prepayment_features(df):
    '''
    Признаки сгенерированные из Стоимость, Внесена предоплата c учетом Ночей, Гостей, номеров:
    - Стоимость за ночь
    - Средняя стоимость номера из номеров
    - Средняя стоимость номера на человека
    - Отношение предоплаты ко стоимости
    - Предоплата на гостя
    - Доля остатка внесения
    - Флаги наличия предоплаты
    '''
    # Стоимость за ночь
    df['cost_per_night'] = df['Стоимость'] / df['Ночей']
    df['cost_per_room'] = df['cost_per_night'] / df['Номеров']
    df['cost_per_guest'] = df['cost_per_room'] / df['Гостей']
    # Отношение предоплаты ко стоимости
    df['prepayment_ratio'] = df['Внесена предоплата'] / df['Стоимость']
    df['remaining_payment_ratio'] = (df['Стоимость'] - df['Внесена предоплата']) / df['Стоимость']
    # Предоплата на гостя
    df['prepayment_per_guest'] = df['Внесена предоплата'] / df['Гостей']
    # Флаги наличия предоплаты
    df['has_prepayment'] = df['Внесена предоплата'].apply(lambda x: 1 if x > 0 else 0)
    # Средняя стоимость на человека
    df['avg_cost_per_person'] = df['Стоимость'] / df['Гостей']
    df['avg_remaining_payment_per_person'] = (df['Стоимость'] - df['Внесена предоплата']) / df['Гостей']

    return df

def log_prepayment_features(df):
    '''Логарифмы взамен оригинала'''
    df['log_total_cost'] = np.log1p(df['Стоимость'])
    df['log_prepayment'] = np.log1p(df['Внесена предоплата'])
    df['log_cost_per_night'] = np.log1p(df['cost_per_night'])
    df['log_cost_per_room'] = np.log1p(df['cost_per_room'])
    df['log_cost_per_guest'] = np.log1p(df['cost_per_guest'])
    df['log_prepayment_per_guest'] = np.log1p(df['prepayment_per_guest'])
    df['log_avg_cost_per_person'] = np.log1p(df['avg_cost_per_person'])
    df['log_avg_remaining_payment_per_person'] = np.log1p(df['avg_remaining_payment_per_person'])

    drop_columns = ['Стоимость','Внесена предоплата','cost_per_night','cost_per_room',
                    'cost_per_guest','prepayment_per_guest','avg_cost_per_person',
                    'avg_remaining_payment_per_person']

    return df.drop(columns=drop_columns)

def guest_features(df):
    '''
    Признаки сгенерированные по 'Гостей' и 'Ночей':
    - флаги размера группы
    - флаг длительности прибывания
    - флаг сути отдыха
    '''
    # флаги размера группы
    df['is_family'] = df['Гостей'].map(lambda x: 1 if x > 1 and x <= 4 else 0)
    df['is_group'] = df['Гостей'].map(lambda x: 1 if x > 4 else 0)
    # флаг сути отдыха
    df['is_weekend'] = (df['checkin_is_weekend'] & (df['checkin_to_checkout_days'] < 3)).astype(int)
    df['is_business_trip'] = ((df['checkin_is_weekend'] == 0) & (df['checkout_is_weekend'] == 0) & (df['checkin_to_checkout_days'] < 7))
    # флаг длительности прибывания
    df['is_long_trip'] = (df['checkin_to_checkout_days'] >= 7).astype(int)
    # взаимодействие между признаками
    df['rooms_times_nights'] = df['Номеров'] * df['Ночей']
    df['guests_per_room'] = df['Гостей'] / df['Номеров']
    df['rooms_per_night'] = df['Номеров'] / df['Ночей']
    df['guests_times_nights'] = df['Гостей'] * df['Ночей']
    df['guests_times_rooms'] = df['Гостей'] * df['Номеров']
    return df

#######################################################################################################################################################################################
# признаки чистики источника бронирования и оплаты
def source_payment_features(df):
    '''Обработать признак источника бронирования'''
    source_map_dict = {
            'Bronevik.com(new)' : 'Bronevik.com',
            'booking.com' : 'booking.com',
            'booking.com (Booking.com)' : 'booking.com',
            '''Acase.ru (ООО "АКАДЕМ-ОНЛАЙН")''' : 'Acase.ru',
            'Acase.ru (ООО "ПРАНДИУМ")' : 'Acase.ru',
            'Alean.ru (13.10.2023-02.06.2025)' : 'Alean.ru',
            'Alean.ru (03.02.2022-31.01.2023)' : 'Alean.ru',
            'Acase.ru (ООО "КАЛЕЙДОСКОП")' : 'Acase.ru',
            'Alean.ru (16.01.2023-06.09.2024)' : 'Alean.ru',
            'Bronevik.com/Bro.Online' : 'Bronevik.com',
            'Alean.ru (31.10.2023-22.06.2025)' : 'Alean.ru',
            'Alean.ru (20.01.2021-31.01.2022)' : 'Alean.ru'
            }
    df['Источник'] = df['Источник'].map(lambda x: source_map_dict.get(x,x))

    return df

def type_payment_features(df):
    '''Обработать признак Способ оплаты'''
    acuiring_bank_map_dict = {
        'Система быстрых платежей: Эквайринг ComfortBooking (Система быстрых платежей)': ['ComfortBooking','sbp'],
        'Банк. карта [Кешбэк. МИР]: Эквайринг TravelLine Pro (Банк. карта)' : ['TravelLine Pro','no info'],
        'Банк. карта: Эквайринг ComfortBooking (Банк. карта)' : ['ComfortBooking','no info'],
        'Банк. карта [Кешбэк. МИР]: Эквайринг ComfortBooking (Банк. карта)' : ['ComfortBooking','no info'],
        'Банк. карта (SberPay): Эквайринг ComfortBooking (Банк. карта) (SberPay)' : ['ComfortBooking','sber'],
        'Банк. карта (Yandex Pay): Эквайринг ComfortBooking (Банк. карта) (Yandex Pay)' : ['ComfortBooking','yandex'],
        'Внешняя система оплаты (Оплата наличными)' : ['cash'],
        'Внешняя система оплаты (Банковская карта)' : ['card'],
        'Гарантия банковской картой' : ['card'],
        'Отложенная электронная оплата: Банк Россия (банк. карта)' : ['card'],
        'Банк. карта: Банк Россия (банк. карта)' : ['card'],
    }
    
    df['type_payment'] = df['Способ оплаты'].map(lambda x: acuiring_bank_map_dict.get(x,['no info'])[0])
    df['bank_payment'] = df['Способ оплаты'].map(lambda x: acuiring_bank_map_dict.get(x,['no info'])[-1])
    df['is outernal_payment'] = df['Способ оплаты'].map(lambda x: int('Внешняя' in x))

    return df

#######################################################################################################################################################################################
# признаки дополнительные
def poly_features(df):
    # Полиномиальные признаки (степени 2)
    df['nights_squared'] = df['Ночей'] ** 2
    df['guests_squared'] = df['Гостей'] ** 2
    df['rooms_squared'] = df['Номеров'] ** 2
    if 'Стоимость' in df.columns:
        df['cost_squared'] = df['Стоимость'] ** 2
        df['sqrt_cost'] = np.sqrt(df['Стоимость'])

    # Корень квадратный из признаков
    df['sqrt_nights'] = np.sqrt(df['Ночей'])
    df['sqrt_guests'] = np.sqrt(df['Гостей'])
    df['sqrt_rooms'] = np.sqrt(df['Номеров'])

    return df


#######################################################################################################################################################################################
# сборка признаков

# Функция для расширенного создания признаков
def features_pack_v1(df):
    '''
    Собирает признаки по датам, категории номера, стоимости, гостям, источнику бронирования, источнику оплаты
    '''
    # добавляем регион
    region_map_dict = {1 : 1, 2 : 1, 3: 2, 4: 2}
    df['region'] = df['Гостиница'].map(lambda x: region_map_dict[x])
    # добавляем даты
    df = date_features(df)
    # Создание признаков из 'Категория номера'
    df = category_room_features(df)
    # Признаки из стоиомости и предоплаты
    df = cost_prepayment_features(df) #log_prepayment_features(df)
    # Гостевые фичи
    df = guest_features(df)
    # Источник бронирования
    df = source_payment_features(df)
    # Создание признаков из 'Способ оплаты'
    df = type_payment_features(df)
    
    return df

# Функция для расширенного создания признаков
def features_pack_v2(df):
    '''
    Собирает признаки по датам, категории номера, стоимости, гостям, источнику бронирования, источнику оплаты
    '''
    # добавляем регион
    region_map_dict = {1 : 1, 2 : 1, 3: 2, 4: 2}
    df['region'] = df['Гостиница'].map(lambda x: region_map_dict[x])
    # добавляем даты
    df = date_features(df)
    # Создание признаков из 'Категория номера'
    df = category_room_features(df)
    # Признаки из стоиомости и предоплаты
    df = log_prepayment_features(cost_prepayment_features(df))
    # Гостевые фичи
    df = guest_features(df)
    # Источник бронирования
    df = source_payment_features(df)
    # Создание признаков из 'Способ оплаты'
    df = type_payment_features(df)
    # поли фичи
    df = poly_features(df)
    
    return df

# Функция для расширенного создания признаков
def features_pack_v3(df, save_file_name_begin=''):
    '''
    Собирает признаки по датам, категории номера, стоимости, гостям, источнику бронирования, источнику оплаты + лаговые агрегаты
    '''
    # добавляем регион
    region_map_dict = {1 : 1, 2 : 1, 3: 2, 4: 2}
    df['region'] = df['Гостиница'].map(lambda x: region_map_dict[x])
    # добавляем даты
    df = date_features(df)
    # Создание признаков из 'Категория номера'
    df = category_room_features(df)
    # Признаки из стоиомости и предоплаты
    df = cost_prepayment_features(df)
    # Гостевые фичи
    df = guest_features(df)
    # Источник бронирования
    df = source_payment_features(df)
    # Создание признаков из 'Способ оплаты'
    df = type_payment_features(df)
    # поли фичи
    df = poly_features(df)
    # Агрегатные фичи
    feature_store_file_name_target = save_file_name_begin + '_' + 'lag_features_for_target'
    if feature_store_file_name_target not in os.listdir():
        calc_mean_target_features(df, 
                                  feature_store_file_name_target, 
                                  group_columns=["Источник",'region'], 
                                  window_sizes=[90, 60, 30, 15], 
                                  min_objects_in_window=1, 
                                  target_col_name='target', date_col_name="Дата бронирования")

    feature_store_file_name_cost = save_file_name_begin + '_' + "lag_features_for_cost"
    if feature_store_file_name_cost not in os.listdir():
        calc_mean_target_features(df, feature_store_file_name_cost, 
                                  group_columns=["Гостиница", "Источник",'region'], 
                                  window_sizes=[90, 60, 30, 15], 
                                  min_objects_in_window=1, 
                                  target_col_name="Стоимость",
                                  date_col_name="Дата бронирования")

    df = add_mean_target_features(df, feature_store_file_name_target, 
                            group_columns=["Источник",'region'], 
                            window_sizes=[90, 60, 30, 15], 
                            date_col_name="Дата бронирования", 
                            target_col_name='target')

    df = add_mean_target_features(df, feature_store_file_name_cost, 
                            group_columns=["Гостиница", 'Источник','region'], 
                            window_sizes=[90, 60, 30, 15], 
                            date_col_name="Дата бронирования", 
                            target_col_name="Стоимость")
    
    return df



# Функция для расширенного создания признаков
# Функция для расширенного создания признаков
def features_pack_v4(df):
    # Конвертация дат в формат datetime
    df['Дата бронирования'] = pd.to_datetime(df['Дата бронирования'])
    df['Заезд'] = pd.to_datetime(df['Заезд'])
    df['Выезд'] = pd.to_datetime(df['Выезд'])

    # Создание новых признаков из дат
    df['booking_year'] = df['Дата бронирования'].dt.year
    df['booking_month'] = df['Дата бронирования'].dt.month
    df['booking_day'] = df['Дата бронирования'].dt.day
    df['booking_dayofweek'] = df['Дата бронирования'].dt.dayofweek
    df['booking_weekofyear'] = df['Дата бронирования'].dt.isocalendar().week
    df['booking_is_weekend'] = df['booking_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['booking_hour'] = df['Дата бронирования'].dt.hour
    df['booking_minute'] = df['Дата бронирования'].dt.minute

    df['checkin_year'] = df['Заезд'].dt.year
    df['checkin_month'] = df['Заезд'].dt.month
    df['checkin_day'] = df['Заезд'].dt.day
    df['checkin_dayofweek'] = df['Заезд'].dt.dayofweek
    df['checkin_is_weekend'] = df['checkin_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    df['checkout_year'] = df['Выезд'].dt.year
    df['checkout_month'] = df['Выезд'].dt.month
    df['checkout_day'] = df['Выезд'].dt.day
    df['checkout_dayofweek'] = df['Выезд'].dt.dayofweek
    df['checkout_is_weekend'] = df['checkout_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Разница между датами
    df['booking_to_checkin_days'] = (df['Заезд'] - df['Дата бронирования']).dt.days
    df['checkin_to_checkout_days'] = (df['Выезд'] - df['Заезд']).dt.days
    df['booking_to_checkout_days'] = (df['Выезд'] - df['Дата бронирования']).dt.days

    # Создание признаков из 'Категория номера'
    df['Категория номера'] = df['Категория номера'].astype(str)
    df['room_category_length'] = df['Категория номера'].apply(len)
    df['room_category_words'] = df['Категория номера'].apply(lambda x: len(x.split()))
    df['room_category_unique_chars'] = df['Категория номера'].apply(lambda x: len(set(x)))
    df['room_category_digits'] = df['Категория номера'].apply(lambda x: sum(c.isdigit() for c in x))
    df['room_category_uppercase'] = df['Категория номера'].apply(lambda x: sum(c.isupper() for c in x))
    df['room_category_title_case'] = df['Категория номера'].apply(lambda x: sum(word.istitle() for word in x.split()))
    df['room_category_num'] = df['Категория номера'].str.extract(r'(\d+)').astype(float)  # Исправлено
    df['room_category_num'] = df['room_category_num'].fillna(df['room_category_num'].mean())

    # Дополнительные числовые признаки
    df['cost_per_night'] = df['Стоимость'] / df['Ночей']
    df['cost_per_room'] = df['Стоимость'] / df['Номеров']
    df['cost_per_guest'] = df['Стоимость'] / df['Гостей']
    df['rooms_per_guest'] = df['Номеров'] / df['Гостей']
    df['nights_per_guest'] = df['Ночей'] / df['Гостей']
    df['cost_per_night_per_room'] = df['Стоимость'] / (df['Ночей'] * df['Номеров'])
    df['cost_per_night_per_guest'] = df['Стоимость'] / (df['Ночей'] * df['Гостей'])
    df['prepayment_ratio'] = df['Внесена предоплата'] / df['Стоимость']
    df['remaining_payment'] = df['Стоимость'] - df['Внесена предоплата']
    df['remaining_payment_ratio'] = df['remaining_payment'] / df['Стоимость']

    # Флаги наличия предоплаты
    df['has_prepayment'] = df['Внесена предоплата'].apply(lambda x: 1 if x > 0 else 0)

    # Взаимодействия между признаками
    df['rooms_times_nights'] = df['Номеров'] * df['Ночей']
    df['guests_per_room'] = df['Гостей'] / df['Номеров']

    # Средняя стоимость на человека
    df['avg_cost_per_person'] = df['Стоимость'] / df['Гостей']

    return df




























