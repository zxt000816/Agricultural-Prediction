import torch, json
from math import floor
import pandas as pd
import numpy as np

from FileManager.dataManager import dataManager

from AnalyzeTools.models import autoregressive_integrated_moving_average, linear_regression, support_vector_regression, random_forest, gradient_boosting
from AnalyzeTools.prepare import data_split, model_eval, pathForSavingModels
from AnalyzeTools.preprocess import preprocessData
from AnalyzeTools.superModels import DEEPAR, TFT, RNN

params_path = './Models/single'
train_size = 0.8
product_object = json.load(open("./File information.json", "r", encoding='utf8'))

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
        
        df, input_features = preprocessData(df, 'date', target)
        predictions_x_axis = df['date'][floor(len(df)*train_size):].values

        # prepare dataset for statistics and Macnhine models
        ml_split_params = {'Model': 'ML', 'Future': 1}
        X_train, X_test, y_train, y_test, input_scaler, output_scaler = data_split(df, input_features, output=target, train_size=train_size, scaling=True, **ml_split_params)

        ''' Input data into models and Evaluate model results '''
        ml_searchCV_params = {
            'base_dir': params_path,
            'product': product_and_product_type,
            'attribute': product_attribute,
            'raw': raw_file_name,
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

        data = df.copy()

        data['time_idx'] = range(len(data))
        data['group'] = product

        training_cutoff = floor(len(data) * train_size)

        max_prediction_length = 1
        max_encoder_length = 30 # 7, 14, 30, 60, 120
        batch_size = 64

        group = ['group']
        time_varying_known_categoricals = ['month', 'week']
        time_varying_unknown_categoricals = []
        time_varying_known_reals = ['time_idx']
        time_varying_unknown_reals = input_features + [target]

        print("\nLSTM")
        lstm, val_dataloader = RNN(
            data,
            training_cutoff,
            target,
            group,
            max_encoder_length,
            max_prediction_length,
            time_varying_known_categoricals,
            time_varying_unknown_categoricals,
            time_varying_known_reals,
            batch_size,
            pathForSavingModels(product_and_product_type, product_attribute, raw_file_name, 'LSTM'),
            'LSTM'
        )

        # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        # lstm_predictions = lstm.predict(val_dataloader)
        # model_eval(actuals, lstm_predictions, predictions_x_axis, stdout=True, vis=True)

        print("\nGRU")
        gru, val_dataloader = RNN(
            data,
            training_cutoff,
            target,
            group,
            max_encoder_length,
            max_prediction_length,
            time_varying_known_categoricals,
            time_varying_unknown_categoricals,
            time_varying_known_reals,
            batch_size,
            pathForSavingModels(product_and_product_type, product_attribute, raw_file_name, 'GRU'),
            'GRU'
        )

        # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        # gru_predictions = gru.predict(val_dataloader)
        # model_eval(actuals, gru_predictions, predictions_x_axis, stdout=True, vis=True)

        print("\nDeepAR")
        deep_ar, val_dataloader = DEEPAR(
            data,
            training_cutoff,
            target,
            group,
            max_encoder_length,
            max_prediction_length,
            time_varying_known_categoricals,
            time_varying_unknown_categoricals,
            time_varying_known_reals,
            batch_size,
            pathForSavingModels(product_and_product_type, product_attribute, raw_file_name, 'DEEPAR'),
        )

        # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        # deepar_predictions = deep_ar.predict(val_dataloader)

        # model_eval(actuals, deepar_predictions, predictions_x_axis, stdout=True, vis=True)

        print("\nTFT")
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
            pathForSavingModels(product_and_product_type, product_attribute, raw_file_name, 'TFT'),
        )

        # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
        # tft_predictions = tft.predict(val_dataloader)
        # model_eval(actuals, tft_predictions, predictions_x_axis, stdout=True, vis=True)
    except BaseException as err:
        err_logs[f'{i+1}/{num_experiments}__{experiment}'] = str(err)

with open('error_logs.json', 'w', encoding='utf8') as f:
    json.dump(err_logs, f, ensure_ascii=False)


