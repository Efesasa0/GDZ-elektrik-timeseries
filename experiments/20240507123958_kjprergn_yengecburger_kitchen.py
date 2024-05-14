# %%
from yengecburger_v4 import cook_yengecburger
import numpy as np
from datetime import datetime
import random
import string
import os
import shutil

experiment_hash = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + ''.join(random.choices(string.ascii_lowercase, k=8))

config_params = {
    "WEATHER_ROLL_TYPES": ["mean", "max", "min"],
    "WEATHER_BACKWARD_ROLL": [3, 7, 15],
    "WEATHER_FORWARD_ROLL": [3, 7, 15],
    "LAG_FEATURES": ["target", "bildirimli_sum"],
    "LAGS": [1, 2, 3],
    "NEIGHBOR_FEATURES": ["target", "bildirimli_sum"],
    "HOW_CLOSE_LIST": [1, 2, 3],
    "MODELS_PARAMS": {
        "xgboost": {
            'init': {
                'enable_categorical': True,
                'n_estimators': 3000,
                'learning_rate': 0.01,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': 42,
                'eval_metric': 'mae',
                'early_stopping_rounds': 100
            },
            'fit': {
                'verbose': True,
            }
        },
        "catboost": {
            'init': {
                'iterations': 3000,
                'learning_rate': 0.01,
                'depth': 8,
                'l2_leaf_reg': 5,
                'random_seed': 42
            },
            'fit': {
                'silent': True
            }
        },
        "lightgbm": {
            'init': {
                'num_leaves': 64,
                'learning_rate': 0.01,
                'n_estimators': 3000,
                'min_child_samples': 20,
                'random_state': 42,
                'verbosity': 1
            },
            'fit': {
                
            }
        }
    },
    "model_weights": {'catboost': 0.5, 'xgboost': 0.3, 'lightgbm': 0.2},
    "ENABLE_HOLIDAY": True,
    "ENABLE_NEIGHBOR": True,
    "ENABLE_COORDINATES": True,
    "ENABLE_WEATHER": True,
    "ENABLE_LAG": True,
    "ENABLE_DATETIME": True,
    "VAL_START_DATE": '2023-08-16',
    "PLOT_VALIDATION": True,
    "experiment_folder": 'experiments',
    "experiment_hash": experiment_hash
}


cook_yengecburger(config_params)

# Save the script
shutil.copyfile(os.path.abspath(__file__), 
                f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_{os.path.basename(__file__)}')

# %%
