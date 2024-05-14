#%%
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import Pool, CatBoostRegressor
import lightgbm as lgb
from datetime import datetime
import random
import string
import shutil
import inspect

plt.style.use('fivethirtyeight')

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
# %%

def create_submission(test_df, model_names, filename):
     
    # weighted sum of models
    test_df['weighted_pred'] = test_df.apply(lambda x: np.sum([w * x[f'target_pred_{model}'] for model, w in model_names.items()]), axis=1)
    
    submission = test_df[['item_id','timestamp','weighted_pred']].reset_index().copy()
    submission['unique_id'] = submission['timestamp'].astype(str) + '-' + submission['item_id'].astype(str)
    submission = submission.rename(columns={'weighted_pred':'bildirimsiz_sum'})
    
    submission = submission.drop(columns=['index','item_id','timestamp'])
    submission = submission[['unique_id','bildirimsiz_sum']]
    submission['bildirimsiz_sum'] = submission['bildirimsiz_sum'].round(0).astype(int)
    submission.to_csv(filename, index=False)
    return submission

def union_train_test(train, test):
    train = train.copy()
    test = test.copy()
    out = pd.concat([train, test], axis=0)
    
    return out

def standardize_format(holidays_raw, weather_raw, train_raw, test_raw):
    holidays, weather, train, test = holidays_raw.copy(), weather_raw.copy(), train_raw.copy(), test_raw.copy()
    
    # HOLIDAYS
    holidays['timestamp'] = holidays.apply(lambda r: f'{r["Yıl"]}-{r["Ay"]:02d}-{r["Gün"]:02d}', axis=1)
    holidays['timestamp'] = holidays['timestamp'].astype('datetime64[ns]')
    #holiday_names = list(holidays['Tatil Adı'].unique()).append("no_holiday")
    #holidays['holiday'] = pd.Categorical(holidays['Tatil Adı'], categories=holiday_names)
    holidays['holiday'] = holidays['Tatil Adı'].astype("category")
    holidays['holiday'] = holidays['holiday'].cat.add_categories(["no_holiday"])
    holidays.drop(columns=['Yıl','Ay','Gün', 'Tatil Adı'], inplace=True)
    holidays = holidays.sort_values("timestamp").reset_index(drop=True)
    
    # WEATHER
    weather = weather.rename(columns={s:s.replace(':','') for s in weather.columns})
    weather["timestamp_hourly"] = weather["date"].astype('datetime64[ns]')
    weather['timestamp'] = weather["timestamp_hourly"].dt.date
    weather['timestamp'] = weather['timestamp'].astype('datetime64[ns]')
    weather['item_id'] = weather['name'].str.lower().astype('category')
    weather.drop(columns=['date','name'], inplace=True)
    weather = weather.sort_values(["item_id", "timestamp_hourly"]).reset_index(drop=True)
    
    # TRAIN
    train['timestamp'] = train['tarih'].astype('datetime64[ns]')
    train['item_id'] = train['ilce'].astype('category')
    train['target'] = train['bildirimsiz_sum'].astype('float64')
    train['bildirimli_sum'] = train['bildirimli_sum'].astype('float64')
    train.drop(columns=['tarih','ilce','bildirimsiz_sum'], inplace=True)
    train = train[["item_id", "timestamp", "target", "bildirimli_sum"]]
    train = train.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    
    # TEST
    test['timestamp'] = test['tarih'].astype('datetime64[ns]')
    test['item_id'] = test['ilce'].astype('category')
    test['bildirimli_sum'] = test['bildirimli_sum'].astype('float64')
    test.drop(columns=['tarih','ilce'], inplace=True)
    test = test[["item_id", "timestamp", "bildirimli_sum"]]
    test = test.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    
    return holidays, weather, train, test

def prepare_coordinates(weather):
    coordinates = weather[['item_id','lat','lon']].drop_duplicates().copy()
    coordinates = coordinates.sort_values("item_id").reset_index(drop=True)
    
    return coordinates

def prepare_daily_weather(weather):
    ## Convert hourly to daily by mean
    weather_daily = weather.groupby(['item_id','timestamp']).agg('mean', numeric_only=True)
    weather_daily.reset_index(inplace=True)
    weather_daily = weather_daily.sort_values(['item_id','timestamp'])

    # not neeeded in weather
    weather_daily.drop(columns=['lon','lat'], inplace=True)
    
    return weather_daily

def densify(all_data):
    min_date, max_date = all_data["timestamp"].dt.date.min(), all_data["timestamp"].dt.date.max()
    
    all_dates = pd.date_range(min_date, max_date)
    all_items = all_data["item_id"].unique()
    all_combs = [(item_id, timestamp) for item_id in all_items for timestamp in all_dates]
    dense = pd.DataFrame(all_combs, columns=["item_id", "timestamp"])
    dense["item_id"] = dense["item_id"].astype("category")
    dense = dense.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    all_data = dense.merge(all_data, how='left', on=["item_id", "timestamp"])
    
    return all_data

def pre_imputation(all_data, train_start_date, train_end_date, test_start_date, test_end_date):
    
    # TARGET TEST should be 0
    all_data.loc[all_data['timestamp']>=test_start_date, 'target'] = 0.0
    
    # TARGET TRAIN should be interpolate
    all_data['target'] = all_data['target'].interpolate()
    
    # BILDIRIMLI_SUM TRAIN should interpolate
    if 'bildirimli_sum' in all_data.columns:
        all_data['bildirimli_sum'] = all_data['bildirimli_sum'].interpolate()
    
    # HOLIDAY
    if 'holiday' in all_data.columns:
        all_data['holiday'] = all_data['holiday'].fillna("no_holiday")
    
    assert(all_data.isna().sum().sum() == 0)
    
    return all_data

def create_holiday_features(all_data, holiday_dict=None, holiday_in_n_days_ls=None):   
    
    item_ls = list(all_data["item_id"].unique())
    if holiday_dict is None:
        holiday_dict = {'dini':["RAMADAN", 'SACRIFIC'],
                        'ramazan':['RAMADAN'],
                        'kurban':['SACRIFIC'],
                        'yilbasi':['NEW YEA'],
                        'resmi':['REPUBLIC', 'VICTORY', 'NATIONAL', 'LABOUR', 'SPORT']}
    if holiday_in_n_days_ls is None:
        holiday_in_n_days_ls = [3, 7, 15]
        
    for holiday_name, holiday_keys in holiday_dict.items():
        all_data[holiday_name] = all_data['holiday'].apply(lambda x: 1 if any(word.upper() in x.upper() for word in holiday_keys) else 0)
        all_data[holiday_name] = all_data[holiday_name].astype(int).astype('category') 
        
        for item_id in item_ls:
            for i in holiday_in_n_days_ls:
                temp_index = all_data['item_id'] == item_id
                temp = all_data[temp_index]
                all_data.loc[temp_index, f"is_{holiday_name}_in_next_{i}_days"] = temp[holiday_name].rolling(i, min_periods=1).max().shift(-i).fillna(0.0).astype(int).astype('category')
    return all_data

def create_weather_features(all_data, columns, roll_types=None, rolls_forward=None, rolls_backward=None):
    
    if roll_types is None:
        roll_types = ["mean", "max", "min"]
    if rolls_forward is None:
        rolls_forward = [1, 2, 3]
    if rolls_backward is None:
        roll_backward = [2, 3, 4]
    
    item_ls = list(all_data["item_id"].unique())
    
    for item_id in item_ls:
        temp_index = all_data['item_id'] == item_id
        for col in columns:
            temp = all_data[temp_index][col]
            for roll_type in roll_types:
                for roll in rolls_backward:
                    temp_rolled = temp.rolling(roll, min_periods=1).agg(roll_type)
                    all_data.loc[temp_index, f"{col}_broll_{roll}_{roll_type}"] = temp_rolled
                for roll in rolls_forward:
                    temp_rolled = temp.shift(-roll).rolling(roll, min_periods=1).agg(roll_type).ffill()
                    all_data.loc[temp_index, f"{col}_froll_{roll}_{roll_type}"] = temp_rolled
    
    return all_data

def create_datetime_features(all_data):
    all_data['month'] = all_data['timestamp'].dt.month.astype('int').astype('category')
    all_data['year'] = all_data['timestamp'].dt.year.astype('int').astype('category')
    all_data['dayofweek'] = all_data['timestamp'].dt.dayofweek.astype('int').astype('category')
    all_data['quarter'] = all_data['timestamp'].dt.quarter.astype('int').astype('category')
    all_data['dayofmonth'] = all_data['timestamp'].dt.day.astype('int').astype('category')
    all_data['weekofyear'] = all_data['timestamp'].dt.isocalendar()['week'].astype('int').astype('category')
    
    return all_data

def create_lag_features(all_data, columns, lags):
    item_ls = list(all_data["item_id"].unique())
    
    for item_id in item_ls:
        temp_index = all_data['item_id'] == item_id
        for col in columns:
            temp = all_data[temp_index][col]
            for lag in lags:
                temp_lag = temp.shift(lag).bfill()
                all_data.loc[temp_index, f'lag_{lag}_{col}'] = temp_lag
                
    return all_data

def create_roll_features(all_data, columns, rolls, roll_types):
    item_ls = list(all_data["item_id"].unique())

    for item_id in item_ls:
        temp_index = all_data['item_id'] == item_id
        for col in columns:
            temp = all_data[temp_index][col]
            for roll in rolls:
                for roll_type in roll_types:
                    temp_rolled = temp.rolling(roll, min_periods=1).agg(roll_type)
                    all_data.loc[temp_index, f'{col}_roll_{roll}_{roll_type}'] = temp_rolled

    return all_data 

def create_distance_dict(all_data):
    items_ls = sorted(list(all_data['item_id'].unique()))
    n = len(items_ls)
    dist_dict = dict()
    dist_mat = np.zeros((n,n))
    nearest = dict()
    
    for i, source in enumerate(items_ls):
        dist_dict[source] = dict()
        nearest[source] = dict()
        source_coords = all_data[all_data['item_id'] == source][['lat','lon']].to_numpy()[0]
        
        for j, target in enumerate(items_ls):
            target_coords = all_data[all_data['item_id'] == target][['lat','lon']].to_numpy()[0]
            distance = np.sum(np.square(source_coords-target_coords))
            dist_dict[source][target] = distance
            dist_mat[i,j] = distance
        nearest_indices = np.argsort(dist_mat[i])
        for k, nearest_index in enumerate(nearest_indices):
            nearest[source][k] = items_ls[nearest_index]
            
    return dist_dict, dist_mat, nearest, items_ls

def copy_features_from_neighbor(all_data, features, how_close_list, nearest_dict):
    self_item_id = all_data.index.get_level_values('item_id')
    items = list(self_item_id.unique())
    for feature in features:    
        for how_close in how_close_list:
            col = f'nearest_{how_close}_{feature}'
            if feature == 'item_id':                
                all_data[col] = pd.Categorical(self_item_id, categories=items)
            else:
                all_data[col] = 0.0

    for feature in features:
        for item in items:
            for how_close in how_close_list:
                col = f'nearest_{how_close}_{feature}'
                neighbor = nearest_dict[item][how_close]
                # Get values from neighbor
                if feature == 'item_id':
                    neighbor_values = neighbor
                else:
                    neighbor_values = all_data.loc[(neighbor, slice(None)),feature].to_list()
                all_data.loc[(item,slice(None)),col] = neighbor_values

    return all_data

def create_neighbor_features(all_data, coordinates, columns, proximity):
    
    dist_dict, dist_mat, nearest, items = create_distance_dict(coordinates)
    # for neighbor features, multiindex is required
    all_data = all_data.set_index(['item_id','timestamp'])

    all_data = copy_features_from_neighbor(all_data, columns, proximity, nearest)

    # remote multiindex again
    all_data = all_data.reset_index()
    
    return all_data

def validation_prep(df):
    item_ls = list(df['item_id'].unique())
    
    # nearest_n_target[:]  --> 0
    pattern = re.compile(r'^nearest_([0-9+])_target$')
    nearest_columns = []
    for col in df.columns:
        match = pattern.match(col)
        if match:
            nearest_columns.append(col)
    for item_id in item_ls:
        index = df['item_id'] == item_id
        for col in nearest_columns:
            df.loc[index, col] = 0.0
            
    # lag_n_target[n:] -->  0
    pattern = re.compile(r'^lag_([0-9+])_target$')
    lag_target_columns = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            lag_target_columns[col] = int(match.group(1))
    for item_id in item_ls:
        index = df['item_id'] == item_id
        for col, lag in lag_target_columns.items():
            temp = df.loc[index, col].to_numpy()
            temp[lag:] = 0.0
            df.loc[index, col] = temp
    
    # nearest_m_lag_n_target[n:] -->  0
    pattern = re.compile(r'^nearest_[0-9]+_lag_([0-9+])_target$')
    nearest_lag_target_columns = {}
    for col in df.columns:
        match = pattern.match(col)
        if match:
            nearest_lag_target_columns[col] = int(match.group(1))
    for item_id in item_ls:
        index = df['item_id'] == item_id
        for col, lag in nearest_lag_target_columns.items():
            temp = df.loc[index, col].to_numpy()
            temp[lag:] = 0.0
            df.loc[index, col] = temp
            
    return df

def split_all_data(all_data, val_start_date, test_start_date, horizon):
    val_range = pd.date_range(val_start_date, periods=horizon, freq="D")

    train = all_data.query(f'timestamp < "{val_start_date}"').copy()
    val = all_data[all_data['timestamp'].isin(val_range)].copy()
    test = all_data.query(f'timestamp >= "{test_start_date}"').copy()
    
    val = validation_prep(val)
    return train, val, test

def create_io_features(all_data, label='target'):
    all_cols = set(all_data.columns)
    feature_cols = list(all_cols - set({'timestamp',label}))
    
    return all_data[feature_cols], all_data[label]

def create_io_pairs(train, val, test):
    X_train, y_train = create_io_features(train, label='target')
    X_val, y_val = create_io_features(val, label='target')
    X_test, y_test = create_io_features(test, label='target')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_feature_importance(models_params, model_dict, val, tst, X_test):
    number_of_models = len(model_dict.keys())
    fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(15,15))
    axes_flat = axes.flatten()
    for ax, model_name in zip(axes.flatten()[:number_of_models], model_dict.keys()):
        feature_importance = model_dict[model_name].feature_importances_
        sorted_idx = np.argsort(feature_importance)[-20:]

        ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        ax.set_yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
        ax.set_yticklabels(np.array(X_test.columns)[sorted_idx])
        ax.set_title(model_name)
    y_position = 0.0
    for model_name, model_params in models_params.items():
        fig.text(0.7, y_position, f'{model_name}:::{model_params}',
                 fontsize=12,verticalalignment='center', transform=fig.transFigure)
        y_position+=0.05
    axes_flat[3].axis('off')

    plt.tight_layout(rect=[0.2, 0, 1, 0.90])
    plt.subplots_adjust(top=0.95)  # Adjust the top to make room for the suptitle
    plt.show()
    
def evaluate_validation(model_dict, val, plot_validation, file=None):
    for model_name in model_dict.keys():
        mae_score = np.mean(np.abs(val['target']-val[f'target_pred_{model_name}']))
        model_dict[model_name]['val_mae'] = mae_score

        model_dict[model_name]['val_mae_by_item_id'] = dict()
        for item_id in val['item_id'].unique():
            mae_score = np.mean(np.abs(val[val['item_id']==item_id]['target']-val[val['item_id']==item_id][f'target_pred_{model_name}']))
            model_dict[model_name]['val_mae_by_item_id'][item_id] = mae_score

    if plot_validation:
        fig, ax=plt.subplots(figsize=(15,5))
        sns.lineplot(x='timestamp',y='target',data=val, ax=ax, label='target')
        for model_name in model_dict.keys():
            sns.lineplot(x='timestamp',y=f'target_pred_{model_name}',data=val, ax=ax, label=f'{model_name} - {model_dict[model_name]["val_mae"]:5.3f}')
        plt.title('Validation results')
        plt.xlabel('bildirimsiz_sum')
        if file is not None:
            plt.savefig(f'{file}')
        #plt.show()
    return model_dict

def plot_test_predictions(model_dict, tst):
    fig, ax=plt.subplots(figsize=(15,5))
    for model_name in model_dict.keys():
        sns.lineplot(x='timestamp',y=f'target_pred_{model_name}',data=tst, ax=ax, label=f'{model_name}')
    plt.title(f'Test predictions')
    plt.ylabel('bildirimsiz_sum')
    plt.savefig('test_plot.png')
    plt.show()

    
def make_forecast(df, model_name, model_dict, max_horizon, cat_features=None):
    X = df.copy()
    model = model_dict[model_name]['model']
    # Find the lag target columns 
    pattern = re.compile(r'^lag_([0-9+])_target$')
    lag_target_columns = {}
    for col in X.columns:
        match = pattern.match(col)
        if match:
            lag_target_columns[col] = int(match.group(1))
            
    # nearest_m_lag_n_target columns
    pattern = re.compile(r'^nearest_([0-9]+)_lag_([0-9+])_target$')
    nearest_lag_target_columns = {}
    for col in X.columns:
        match = pattern.match(col)
        if match:
            nearest_lag_target_columns[col] = (int(match.group(1)),int(match.group(2)))

    # First index to keep track of item_id. 
    X['item'] = X['item_id']
    # Second index to keep track of time
    items = list(X['item_id'].unique())
    item_count = len(items)
    X['horizon'] = np.concatenate([np.arange(max_horizon) for i in range(item_count)])
    # Create a multi-index for easy access to (item, horizon) pairs
    X = X.set_index(['item','horizon'])
    
    # To store the predictions
    y = pd.DataFrame(index=X.index)
    # pred columns is initialized
    y['pred'] = 0

    # Starting from time 0, 1, .., max_horizon-1
    for horizon in range(max_horizon):
        # Make prediction on horizon for all items
        # (slice(None), horizon) --> for all items at horizon
        X_input = X.loc[(slice(None), horizon),:]
        # Catboost uses a custom Pool object for prediction
        if model_name == 'catboost':
            X_input = Pool(X_input, cat_features=cat_features)
        pred = model.predict(X_input)
        # and store the result in y['pred']
        y.loc[(slice(None), horizon), 'pred'] = pred
        
        pred_dict = {i: p for i, p in zip(items, pred)}

        # Forward propagate prediction to lag target columns
        # target at horizon --> copy --> lag_2_target at horizon + 2
        for col, lag in lag_target_columns.items():
            if horizon + lag < max_horizon:
                X.loc[(slice(None), horizon + lag), col] = pred
                
        # Forward propagate prediction to nearest_m_lag_n_target
        # target at horizon --> copy --> lag_2_target at horizon + 2
        for col, (nearest,lag) in nearest_lag_target_columns.items():
            pred_at_nearest = [pred_dict[nearest_item] for nearest_item in X.loc[(slice(None),horizon),f'nearest_{nearest}_item_id'].values]
            if horizon + lag < max_horizon:
                X.loc[(slice(None), horizon + lag), col] = pred_at_nearest  

    return y['pred'].values
    
def train_predict(all_data, X_train, y_train, X_val, y_val, X_test, val, tst, config_params,
                  print_feature_importance=False,
                  plot_validation=False,
                  print_test_predictions=False,
                  max_horizon=29):
    
    models_params = config_params['MODELS_PARAMS']
    categorical_columns = all_data.select_dtypes(include=['object', 'category']).columns.tolist()
    model_dict = dict()
    for model_name, params in models_params.items():
        #XGBOOST
        if model_name=="xgboost":
            model_dict[model_name] = dict()
            model_dict[model_name]['model'] = xgb.XGBRegressor(**params['init'])
            model_dict[model_name]['model'].fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], **params['fit'])
            val['target_pred_xgboost'] = make_forecast(X_val, model_name, model_dict, max_horizon=max_horizon) 
            tst['target_pred_xgboost'] = make_forecast(X_test, model_name, model_dict, max_horizon=max_horizon) 
            
        # CATBOOST
        if model_name=="catboost":            
            train_pool = Pool(X_train,y_train, cat_features=categorical_columns)
            val_pool = Pool(X_val,y_val, cat_features=categorical_columns)
            test_pool = Pool(X_test, cat_features=categorical_columns)
            model_dict[model_name] = dict()
            model_dict[model_name]['model']  = CatBoostRegressor(**params['init'])
            model_dict[model_name]['model'].fit(train_pool,**params['fit'])
            val['target_pred_catboost'] = make_forecast(X_val, model_name, 
                                                        model_dict, 
                                                        max_horizon=max_horizon,
                                                        cat_features=categorical_columns
                                                       ) 
            tst['target_pred_catboost'] = make_forecast(X_test, model_name, 
                                                        model_dict, 
                                                        max_horizon=max_horizon,
                                                        cat_features=categorical_columns)   
            
        # LIGHTGBM
        if model_name=="lightgbm":
            model_dict[model_name] = dict()
            model_dict[model_name]['model']  = lgb.LGBMRegressor(**params['init'])
            model_dict[model_name]['model'].fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],**params['fit'])
            val['target_pred_lightgbm'] = make_forecast(X_val, model_name, model_dict, max_horizon=max_horizon) 
            tst['target_pred_lightgbm'] = make_forecast(X_test, model_name, model_dict, max_horizon=max_horizon)
    
    if print_feature_importance:
        plot_feature_importance(models_params, model_dict, val, tst, X_test)

    start_of_val = val['timestamp'].min()
    # Format as YYYY-MM-DD
    start_of_val = start_of_val.strftime('%Y-%m-%d')
    val_plot_file = f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_val_plot_{start_of_val}.png'
    model_dict = evaluate_validation(model_dict, val, plot_validation, file=val_plot_file)
    if print_test_predictions:
        plot_test_predictions(model_dict, tst)
    return model_dict, val, tst
# %%
def read_datas():
    holidays_raw = pd.read_csv("kaggle/input/gdz-elektrik-datathon-2024/holidays.csv")
    weather_raw = pd.read_csv("kaggle/input/gdz-elektrik-datathon-2024/weather.csv")
    train_raw = pd.read_csv("kaggle/input/gdz-elektrik-datathon-2024/train.csv")
    test_raw = pd.read_csv("kaggle/input/gdz-elektrik-datathon-2024/test.csv")

    return holidays_raw, weather_raw, train_raw, test_raw
# %%
def prepare_datas(holidays_raw, weather_raw, train_raw, test_raw):
    holidays, weather, train, test = standardize_format(holidays_raw, weather_raw, train_raw, test_raw)
    
    return holidays, weather, train, test
# %%
def cook_yengecburger(config_params):

    print('Writing results to ', config_params['experiment_folder'])
    # Create folder 'exp' if not exists
    if not os.path.exists(config_params['experiment_folder']):
        os.makedirs(config_params['experiment_folder'])

    holidays_raw, weather_raw, train_raw, test_raw = read_datas()
    holidays, weather, train, test = prepare_datas(holidays_raw, weather_raw, train_raw, test_raw)

    train_start_date = train["timestamp"].min()
    train_end_date = train["timestamp"].max()

    test_start_date = test["timestamp"].min()
    test_end_date = test["timestamp"].max()
    weather_features = list(set(weather.columns) - set(["item_id", "timestamp", "lat", "lon","timestamp_hourly"]))

    HORIZON = 29

    weather_daily = prepare_daily_weather(weather)
    coordinates = prepare_coordinates(weather)

    # DATA PREPERATION
    all_data = union_train_test(train, test)
    all_data = densify(all_data)
    if config_params["ENABLE_HOLIDAY"]:
        all_data = all_data.merge(holidays, on="timestamp", how="left")
    if config_params["ENABLE_WEATHER"]:
        all_data = all_data.merge(weather_daily, on=["item_id", "timestamp"], how="left")
    if config_params["ENABLE_COORDINATES"]:
        all_data = all_data.merge(coordinates, on="item_id", how="left")

    # DATA IMPUTATION
    all_data = pre_imputation(all_data,
                            train_start_date,
                            train_end_date,
                            test_start_date,
                            test_end_date)

    # FEATURE ENGINEERING
    # TODO: holiday long consecutive n days ::: 
    if config_params["ENABLE_HOLIDAY"]:
        all_data = create_holiday_features(all_data)
    if config_params["ENABLE_WEATHER"]:
        all_data = create_weather_features(all_data,
                                        weather_features,
                                        config_params["WEATHER_ROLL_TYPES"],
                                        config_params["WEATHER_FORWARD_ROLL"],
                                        config_params["WEATHER_BACKWARD_ROLL"])
    if config_params["ENABLE_DATETIME"]:
        all_data = create_datetime_features(all_data)
    if config_params["ENABLE_LAG"]:
        all_data = create_lag_features(all_data,
                                    config_params["LAG_FEATURES"],
                                    config_params["LAGS"])
    if config_params["ENABLE_ROLL"]:
        all_data = create_roll_features(all_data,
                                    config_params["ROLL_FEATURES"],
                                    config_params["ROLLS"],
                                    config_params["ROLL_TYPES"])
    if config_params["ENABLE_NEIGHBOR"]:
        all_data = create_neighbor_features(all_data,
                                            coordinates,
                                            config_params["NEIGHBOR_FEATURES"],
                                            config_params["HOW_CLOSE_LIST"])
    # MACHINE LEARNING

    model_names = list(config_params["MODELS_PARAMS"].keys())
    val_results = pd.DataFrame(columns=["item_id","val_start_date"] + model_names)
    val_results = val_results.set_index(["item_id","val_start_date"])
    items = list(all_data["item_id"].unique())
    
    for val_start_date in config_params["VAL_START_DATES"]:


        trn, val, tst = split_all_data(all_data,
                                    val_start_date,
                                    test_start_date,
                                    HORIZON)
        X_train, y_train, X_val, y_val, X_test, y_test = create_io_pairs(trn, val, tst)
        model_dict, val, tst = train_predict(all_data, 
                                            X_train, y_train, X_val, y_val, X_test, 
                                            val, tst,
                                            config_params,
                                            print_feature_importance=False,
                                            plot_validation=config_params["PLOT_VALIDATION"],
                                            print_test_predictions=False,
                                            max_horizon=HORIZON)
        for model_name, model_results in model_dict.items():
            print(f"val: {val_start_date} model: {model_name:14s} validation MAE (agg by item_id): {model_results['val_mae']:5.3f}")

        for item in items:
            for model_name in model_names:
                mae = model_dict[model_name]['val_mae_by_item_id'][item]
                val_results.loc[(item, val_start_date), model_name] = mae
        
        # calculate the average MAE column-wise
    val_results['avg_val_mae_by_model'] = val_results[model_names].mean(axis=1)
    val_results = val_results.sort_values(['avg_val_mae_by_model'])
    val_results = val_results.reset_index()
    val_results.to_excel(f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_val_scores.xlsx')
    
    global_val_mae = val_results[model_names].mean(axis=1).mean(axis=0)
    print('Global MAE average (by item_id, val_date, model) ', global_val_mae)
    with open(f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_global_val_mae.txt', 'w') as f:
        f.write(str(global_val_mae))

    #print(set(all_data.columns))
    #print("#"*200)
    #submission = create_submission(tst, 
    #                config_params["model_weights"], 
    #                filename=f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_submission.csv')
    
    


    lib = inspect.getmodule(create_submission)
    # Find the path of the module mm
    lib_abspath = os.path.abspath(lib.__file__)
    # get the basename of the src
    lib_basename = os.path.basename(lib.__file__)
    shutil.copyfile(lib_abspath, f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_{lib_basename}')

   