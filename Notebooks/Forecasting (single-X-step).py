import torch, json
from math import floor
import pandas as pd
import numpy as np

from FileManager.dataManager import dataManager

from AnalyzeTools.models import autoregressive_integrated_moving_average, linear_regression, support_vector_regression, random_forest, gradient_boosting
from AnalyzeTools.prepare import data_split, model_eval, pathForSavingModels
from AnalyzeTools.preprocess import preprocessData, removeOutliers, createPeriodData
from AnalyzeTools.superModels import DEEPAR, TFT, RNN

predict_type = 'single'
params_path = './Models'
test_size = 0.2
product_object = json.load(open("./File information.json", "r", encoding='utf8'))

period_steps = ['Day-3', 'Day-7', 'Day-15', 'Week-2', 'Week-3', 'Month-2', 'Month-3']

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

            data, input_features = preprocessData(data, 'date', target)
            # predictions_x_axis = data['date'][floor(len(data)*train_size):].values

            # prepare dataset for statistics and Macnhine models
            ml_split_params = {'Model': 'ML', 'Future': max_prediction_length}
            X_train, X_test, y_train, y_test, input_scaler, output_scaler = data_split(df, input_features, target, test_size, scaling=True, **ml_split_params)

            ml_searchCV_params = {
                'base_dir': params_path,
                'product_and_product_type': product_and_product_type,
                'attribute': product_attribute,
                'raw': raw_file_name,
                'predict_type': predict_type,
                'period': period,
                'step': max_prediction_length,
                'save': True
            }
            # stdout = False
            # vis = False

            # print("\nARIMA")
            # arima_predictions = autoregressive_integrated_moving_average(y_train, y_test)
            # model_eval(y_test, arima_predictions, predictions_x_axis, stdout=stdout, vis=vis, **{'scaler': output_scaler})

            print("\nLinear Regression")
            lr, _ = linear_regression(X_train, y_train)
            # lr_predictions = lr.predict(X_test)
            # model_eval(y_test, lr_predictions, predictions_x_axis, stdout=stdout, vis=vis, **{'scaler': output_scaler})

            print("\nSupport Vector Regression")
            svr, _ = support_vector_regression(X_train, y_train, search=True, **ml_searchCV_params)
            # svr_predictions = svr.predict(X_test)
            # model_eval(y_test, svr_predictions, predictions_x_axis, stdout=stdout, vis=vis, **{'scaler': output_scaler})

            print("\nRandom Forest")
            rf, _ = random_forest(X_train, y_train, search=True, **ml_searchCV_params)
            # rf_predictions = rf.predict(X_test)
            # model_eval(y_test, rf_predictions, predictions_x_axis, stdout=stdout, vis=vis, **{'scaler': output_scaler})

            print("\nGradient Boosting")
            gb, _ = gradient_boosting(X_train, y_train, search=True, **ml_searchCV_params)
            # gb_predictions = gb.predict(X_test)
            # model_eval(y_test, gb_predictions, predictions_x_axis, stdout=stdout, vis=vis, **{'scaler': output_scaler})

    except BaseException as err:
        err_logs[f'{i+1}/{num_experiments}__{experiment}'] = str(err)

with open('error_logs(single-X).json', 'w', encoding='utf8') as f:
    json.dump(err_logs, f, ensure_ascii=False)


