# %%
from yengecburger_v4 import cook_yengecburger
import numpy as np
from datetime import datetime
import random
import string
import os
import shutil
import contextlib


experiment_hash = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + ''.join(random.choices(string.ascii_lowercase, k=8))

model_presets = dict()
model_presets['efe'] = {
         "xgboost": {
            'init': {
                'enable_categorical': True,
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'early_stopping_rounds': 50,
                'eval_metric': 'rmse'
            },
            'fit': {
                'verbose': False,
            }
        },
        "catboost": {
            'init': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42
            },
            'fit': {
                'silent': True
            }
        },
        "lightgbm": {
            'init': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'random_state': 42,
                'depth': 6,
                'verbosity': -1,
            },
            'fit': {
            }
        },
    }
model_presets['default'] = {
         "xgboost": {
            'init': {
                'enable_categorical': True,
            },
            'fit': {
                'verbose': False,
            }
        },
        "catboost": {
            'init': {
            },
            'fit': {
                'silent': True
            }
        },
        "lightgbm": {
            'init': {
                'verbosity': -1,
            },
            'fit': {
            }
        },
    }

weather_cols = ['effective_cloud_coverp','t_2mC','global_radW','relative_humidity_2mp','wind_dir_10md','wind_speed_10mms','prob_precip_1hp','t_apparentC']
'''
config_params = {
    "WEATHER_ROLL_TYPES": ['mean','max','min'],
    "WEATHER_BACKWARD_ROLL": [2,3,7,30],
    "WEATHER_FORWARD_ROLL": [1],
    "LAG_FEATURES": ['bildirimli_sum'] + weather_cols, 
    "LAGS": [1,2,3,4,5,6,7],
    "ROLL_FEATURES": ['bildirimli_sum'],
    "ROLLS": [7, 60],
    "ROLL_TYPES": ['mean','min','max'],
    "NEIGHBOR_FEATURES": ['bildirimli_sum','lag_1_bildirimli_sum'] +
      weather_cols +
      [f'{col}_froll_1_{rtype}' for rtype in ['mean'] for col in weather_cols],
    "HOW_CLOSE_LIST": [1,2,3,4,5],
    "MODELS_PARAMS": {
         "xgboost": {
            'init': {
                'enable_categorical': True,
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'eval_metric': 'rmse',
                'early_stopping_rounds': 50,
            },
            'fit': {
                'verbose': False,
            }
        },
        "catboost": {
            'init': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42
            },
            'fit': {
                'silent': True
            }
        },
        "lightgbm": {
            'init': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'random_state': 42,
                'verbosity': -1,
            },
            'fit': {
            }
        },
    },
    "model_weights": {'catboost': 0.33, 'xgboost': 0.33, 'lightgbm': 0.33},
    "ENABLE_HOLIDAY": True,
    "ENABLE_NEIGHBOR": True,
    "ENABLE_COORDINATES": True,
    "ENABLE_WEATHER": True,
    "ENABLE_LAG": True,
    "ENABLE_ROLL": True,
    "ENABLE_DATETIME": True,
    "VAL_START_DATES": ['2024-01-01'],
    "PLOT_VALIDATION": True,
    "experiment_folder": './experiments/',
    "experiment_hash": experiment_hash,
}
'''

config_params = {
    "SMOOTH_TARGET": True,
    "SMOOTH_TARGET_WINDOW_SIZE": 7,
    "SMOOTH_TARGET_METHOD": "savgol",
    "SMOOTH_TARGET_UPSAMPLE_MULTIPLIER":  7, 
    "OUTLIER_METHOD": "REMOVE", # None, "CLIP", or "REMOVE"
    "OUTLIER_LIMIT": 25,
    "WEATHER_ROLL_TYPES": ['mean','max','min'],
    "WEATHER_BACKWARD_ROLL": [2,3,7,30],
    "WEATHER_FORWARD_ROLL": [1],
    "LAG_FEATURES": ['bildirimli_sum'] + weather_cols, 
    "LAGS": [1,2,3,4,5,6,7],
    "TARGET_LAGS": [1,30],
    "ROLL_FEATURES": ['bildirimli_sum'],
    "ROLLS": [7, 60],
    "ROLL_TYPES": ['mean','min','max'],
    "STATIC_FEATURES": ['target','bildirimli_sum'],
    "STATIC_FEATURES_SUB_CATEGORIES": [], #['month', 'year', 'dayofweek', 'quarter', 'dayofmonth','weekofyear'],
    "STATIC_FEATURES_AGG_TYPES": ['mean','std'],
    "NEIGHBOR_FEATURES": ['bildirimli_sum'] + [f'lag_{lag}_bildirimli_sum' for lag in [1,2,3,4,5,6,7]] +
      weather_cols +
      [f'{col}_froll_1_{rtype}' for rtype in ['mean'] for col in weather_cols],
    "HOW_CLOSE_LIST": [1,2,3,4,5],
    "FORWARD_LAG_FEATURES": ['bildirimli_sum'] + weather_cols, 
    "FORWARD_LAGS": [1,2,3],
    "MODELS_PARAMS":  model_presets['default'],
    "model_weights": {'catboost': 0.33, 'xgboost': 0.33, 'lightgbm': 0.33},
    "ENABLE_HOLIDAY": True,
    "ENABLE_NEIGHBOR": False,
    "ENABLE_COORDINATES": False,
    "ENABLE_WEATHER": True,
    "ENABLE_ROLL": False,
    "ENABLE_DATETIME": True,
    "VAL_START_DATES": ['2023-10-01','2023-11-01','2023-12-01','2024-01-01'],
    "PLOT_VALIDATION": True,
    "experiment_folder": './experiments/',
    "experiment_hash": experiment_hash,
}


# Redirect stdout to log file
log_file = f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_log.txt'
with open(log_file, "w") as h, contextlib.redirect_stdout(h):

    cook_yengecburger(config_params)

    # Save the script
    shutil.copyfile(os.path.abspath(__file__), 
                    f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_{os.path.basename(__file__)}')


# %%
