import torch, json
from math import floor
import pandas as pd
import numpy as np

from FileManager.dataManager import dataManager
from AnalyzeTools.prepare import model_eval, pathForSavingModels
from AnalyzeTools.preprocess import preprocessData, removeOutliers, createPeriodData
from AnalyzeTools.superModels import DEEPAR, TFT, RNN

predict_type = 'multiple_interval'
params_path = './Models'
product_object = json.load(open("./File information.json", "r", encoding='utf8'))

period_steps = ['Day-1', 'Day-7', 'Day-15', 
                'Week-1', 'Week-2', 'Week-3', 
                'Month-1', 'Month-2', 'Month-3']

all_experiments= []
for product in product_object.keys():
    for raw_file_name in  product_object[product].keys():
        for product_type in product_object[product][raw_file_name]['product_types']:
            for target in product_object[product][raw_file_name]['targets']:
                all_experiments.append([product, raw_file_name, product_type, target])

num_experiments = len(all_experiments)
err_logs = {}
for i, experiment in enumerate(all_experiments):
    try:
        print("*"*50 + f"  {i+1}/{num_experiments}  "  + "*"*50)
        product, raw_file_name, product_type, target = experiment

        df, product_and_product_type, product_attribute = dataManager(raw_file_name, product, product_type, target)

        if len(df) == 0:
            continue
        
        for period_step in period_steps:
            period, max_prediction_length = period_step.split("-")
            max_prediction_length = int(max_prediction_length)

            data = createPeriodData(df, {'date': 'first', 'others': 'mean'}, period, 'date')

            data, input_features = preprocessData(data, 'date', target, period)

            data['time_idx'] = range(len(data))
            data['group'] = product

            training_cutoff = floor(len(data) * 0.8)

            if period == 'Day':
                max_encoder_length = 30
                batch_size = 64 
            elif period == 'Week':
                max_encoder_length = 15
                batch_size = 32
            elif period == 'Month':
                max_encoder_length = 7
                batch_size = 16

            group = ['group']
            time_varying_known_categoricals = ['month', 'week']
            time_varying_known_categoricals = []
            time_varying_unknown_categoricals = []
            time_varying_known_reals = ['time_idx']
            time_varying_unknown_reals = input_features + [target]

            dl_searchCV_params = {
                'base_dir': params_path,
                'product_and_product_type': product_and_product_type,
                'attribute': product_attribute,
                'raw': raw_file_name,
                'predict_type': predict_type,
                'period': period,
                'step': max_prediction_length,
                'save': True
            }

            training_params = {'max_epochs': 50, 'n_trials': 10, 'output_size': 7}
            saving_dir = pathForSavingModels('TFT', **dl_searchCV_params)
            tft, val_dataloader = TFT(
                data,
                training_cutoff,
                target,
                group,
                max_encoder_length,
                max_prediction_length,
                time_varying_unknown_categoricals,
                time_varying_known_categoricals,
                time_varying_known_reals,
                time_varying_unknown_reals,
                batch_size,
                saving_dir,
                predict_type,
                **training_params,
            )
    except BaseException as err:
        err_logs[f'{i+1}/{num_experiments}_{experiment}'] = str(err)
    
with open('error_logs(multiple-steps-interval).json', 'w', encoding='utf8') as f:
    json.dump(err_logs, f, ensure_ascii=False)
