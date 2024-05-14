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
                'n_estimators': 5000,        # Increase the number of estimators
                'learning_rate': 0.005,      # Lower learning rate
                'max_depth': 10,             # Increase depth
                'subsample': 0.8,            # Same subsample ratio
                'colsample_bytree': 0.8,     # Same feature fraction
                'objective': 'reg:squarederror',
                'random_state': 42,
                'eval_metric': 'mae'
            },
            'fit': {
                'verbose': True,
                'early_stopping_rounds': 200,  # Increase patience
            }
        },
        "catboost": {
            'init': {
                'iterations': 5000,           # Increase iterations
                'learning_rate': 0.005,       # Lower learning rate
                'depth': 10,                  # Increase depth
                'l2_leaf_reg': 6,             # Increase regularization
                'random_seed': 42
            },
            'fit': {
                'silent': False
            }
        },
        "lightgbm": {
            'init': {
                'num_leaves': 128,            # Increase number of leaves
                'learning_rate': 0.005,       # Lower learning rate
                'n_estimators': 5000,         # Increase number of estimators
                'min_child_samples': 10,      # Adjust minimum number of samples per leaf
                'random_state': 42,
                'verbose': 1
            },
            'fit': {
            }
        }
    },
    "model_weights": {'catboost': 0.4, 'xgboost': 0.3, 'lightgbm': 0.3},
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
