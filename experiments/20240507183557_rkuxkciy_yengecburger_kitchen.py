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
    "WEATHER_ROLL_TYPES": [],
    "WEATHER_BACKWARD_ROLL": [],
    "WEATHER_FORWARD_ROLL": [],
    "LAG_FEATURES": ['bildirimli_sum','target'], 
    "LAGS": [1,2,3,4,5,6,7],
    "ROLL_FEATURES": ['bildirimli_sum','target'],
    "ROLLS": [7, 30, 90],
    "ROLL_TYPES": ['mean','min','max'],
    "NEIGHBOR_FEATURES": ['roll_target_90_mean','roll_bildirimli_sum_90_mean','roll_target_30_mean','roll_bildirimli_sum_30_mean','roll_target_7_mean','roll_bildirimli_sum_7_mean'],
    "HOW_CLOSE_LIST": [1,2,3,4,5,6,7],
    "MODELS_PARAMS": {
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
        }
    },
    "model_weights": {'catboost': 0.5, 'xgboost': 0.5, 'lightgbm': 0.0},
    "ENABLE_HOLIDAY": False,
    "ENABLE_NEIGHBOR": False,
    "ENABLE_COORDINATES": False,
    "ENABLE_WEATHER": True,
    "ENABLE_LAG": True,
    "ENABLE_ROLL": True,
    "ENABLE_DATETIME": True,
    "VAL_START_DATES": ['2023-10-01','2023-11-01','2023-12-01','2024-01-01'],
    "PLOT_VALIDATION": True,
    "experiment_folder": 'experiments',
    "experiment_hash": experiment_hash,
}

cook_yengecburger(config_params)

# Save the script
shutil.copyfile(os.path.abspath(__file__), 
                f'{config_params["experiment_folder"]}/{config_params["experiment_hash"]}_{os.path.basename(__file__)}')

# %%
