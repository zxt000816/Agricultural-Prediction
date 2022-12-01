import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as R2

import matplotlib.pyplot as plt
import json, os, torch, copy
import numpy as np
from math import floor

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

plot_template = dict(
    layout=go.Layout({
        'font_size': 18,
        'xaxis_title_font_size': 18,
        'yaxis_title_font_size': 18,
    })
)

def model_eval(Y_true, Y_predict, X_axis, stdout=False, vis=False, ret=False, beautiful=True, **params):
    if not params:
        MAPE, RSQUARE=  mape(Y_true, Y_predict), R2(Y_true, Y_predict)
    else:
        scaler = params['scaler']
        Y_true = scaler.inverse_transform(Y_true.reshape(-1, 1))
        Y_predict = scaler.inverse_transform(Y_predict.reshape(-1, 1))
        MAPE, RSQUARE = mape(Y_true, Y_predict), R2(Y_true, Y_predict)
    if stdout:
        print(f"MAPE: {MAPE} R square: {RSQUARE}")
    if vis:
        if beautiful:
            beautiful_vis_reult(Y_true.ravel(), Y_predict.ravel(), X_axis)
        else:
            vis_result(Y_true, Y_predict, X_axis)
    if ret:
        return MAPE, RSQUARE

def vis_result(Y_true, Y_predict, X_axis):
    plt.figure(figsize=(10, 6))
    plt.plot(X_axis, Y_true, label='Actual')
    plt.plot(X_axis, Y_predict, label='Predict')
    plt.legend()
    plt.show()

def beautiful_vis_reult(Y_true, Y_predict, X_axis):
    fig = go.Figure()
    fig.add_scatter(x=X_axis, y=Y_true, mode='lines', name='실제')
    fig.add_scatter(x=X_axis, y=Y_predict, mode='lines', name='예측')
    fig.update_layout(
        template=plot_template,
        xaxis_title="일자",
        yaxis_title="가격", 
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1000,
        height=600
    )
    fig.show()

def best_dl(X_train, y_train, X_test, y_test, _model, **others):
    hidden_sz_ls = others.get('hidden_sz') if others.get('hidden_sz') else [16, 32, 64, 128, 256]
    num_layers_ls = others.get('number_layers') if others.get('number_layers') else [1, 2, 3, 4]
    lr_ls = others.get('lr') if others.get('lr') else [0.001, 0.005, 0.01, 0.05, 0.1]
    num_epochs_ls = others.get('num_epochs') if others.get('num_epochs') else [50, 100, 150, 200, 250, 300, 500]
    num_experiments = others.get('num_experiments') if others.get('num_experiments') else 100
    scaler = others.get('price_scaler')
    device = others.get('device') if others.get('device') else 'cpu'
    
    results = []
    for num_layers in num_layers_ls:
        for hidden_sz in hidden_sz_ls:
            for lr in lr_ls:
                for num_epochs in num_epochs_ls:
                    for experiment_idx in range(num_experiments):
                        model = _model(X_train, y_train, **{
                            'num_layers': num_layers,
                            'hidden_sz': hidden_sz,
                            'lr': lr,
                            'num_epochs': num_epochs,
                            'device': device
                        })
                    
                        with torch.no_grad():
                            MAPE, RSQUARE = model_eval(y_test.to('cpu'), model(X_test.float()).to('cpu'), **{'price_scaler': scaler})
                        
                        results.append([num_layers, hidden_sz, lr, num_epochs, experiment_idx+1, MAPE, RSQUARE])
    
    results = pd.DataFrame(results, columns=["num_layers", "hidden_size", "learning_rate", "num_epochs", "experiment_ids", "MAPE", "R2"])

    best_params = results.groupby(['num_layers', 'hidden_size', 'learning_rate', 'num_epochs'])\
        .agg({'R2': 'mean', 'MAPE': 'mean'})\
        .sort_values('R2', ascending=False).reset_index().head(1)[['num_layers', 'hidden_size', 'learning_rate', 'num_epochs']].to_dict(orient='records')[0]
    ret_best_params = best_params.copy()
    best_params.update({"device": device})
    return _model(X_train, y_train, **best_params), ret_best_params

def save_best_params(best_params, file_path):
    with open(file_path, 'w') as fp:
        json.dump(best_params, fp)

# Version 1    
# def read_best_params(file_path, model_type='NDL'):
#     with open(file_path) as f:
#         model_best_params = json.load(f)
#         if model_type == 'DL':
#             return model_best_params
#         for key, val in model_best_params.items():
#             model_best_params[key] = [val]
#         return model_best_params

def read_best_params(file_path):
    with open(file_path) as f:
        model_best_params = json.load(f)
        return model_best_params

def data_split(dataframe, input_features, output, test_size, scaling=False, **params):
    data = dataframe.copy()
    input_cols = copy.deepcopy(input_features)
    print(data.shape)

    input_scaler, output_scaler = StandardScaler(), StandardScaler()
    if type(test_size) == float:
        test_size = floor(len(data) * test_size) 

    ### Test ###
    if scaling:
        input_scaler.fit(data[input_features][:-1*test_size])
        output_scaler.fit(data[[output]][:-1*test_size])

        data[input_features] = input_scaler.transform(data[input_features])
        data[[output]] = output_scaler.transform(data[[output]])
    ### Test ###

    if params.get('Model') == 'ML':
        target = f'Future_{output}'
        data[target] = data[output].shift(-1 * params.get('Future'))
        data = data[:-1 * params.get('Future')]

        input_cols.append(output)

        print(data.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            data[input_cols].values, 
            data[target].values, 
            test_size=test_size,
            shuffle=False
        )
        print(f"X_train: {X_train.shape} y_train: {y_train.shape} X_test: {X_test.shape} y_test: {y_test.shape}")

        if scaling:
            # X_train, y_train = input_scaler.fit_transform(X_train), output_scaler.fit_transform(y_train.reshape(-1,1))
            # X_test, y_test = input_scaler.transform(X_test), output_scaler.transform(y_test.reshape(-1,1))
            # return X_train, X_test, y_train.ravel(), y_test.ravel(), input_scaler, output_scaler
            return X_train, X_test, y_train, y_test, input_scaler, output_scaler
        return X_train, X_test, y_train, y_test

    if not scaling:
        raise ValueError("When training deep learning models, scaling must be set as True!")

    training_cutoff = params.get('dividing_line')
    device = params.get('device')

    data.iloc[:training_cutoff-1, input_cols] = input_scaler.fit_transform(data.iloc[:training_cutoff-1, input_cols])
    data.iloc[:training_cutoff-1, [output]] = output_scaler.fit_transform(data.iloc[:training_cutoff-1, [output]])

    X, y = make_dataset(data[input_cols].values, data[output].values, params['future'], params['past'])

    X_train, y_train = input_scaler.fit_transform(X_train), output_scaler.fit_transform(y_train)
    X_test, y_test = input_scaler.transform(X_test), output_scaler.transform(y_test)

    X_train, y_train = X[:training_cutoff, :, [-1]], y[:training_cutoff, -1]
    X_test, y_test = X[training_cutoff:, :, [-1]], y[training_cutoff:, -1]
    print(f"X_train_dl: {X_train.shape} y_train_dl: {y_train.shape} X_test_dl: {X_test.shape} y_test_dl: {y_test.shape}")

    X_train, y_train = torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device)
    X_test, y_test= torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device)

    return X_train, X_test, y_train, y_test, input_scaler, output_scaler

def make_dataset(X_arr, y_arr, future, past):
    X, y = [], []
    for i in range(past, len(X_arr)-future):
        X.append(X_arr[i-past: i, :])
        y.append(y_arr[i+future-1, [-1]])
    return np.array(X, dtype=np.float16), np.array(y, dtype=np.float16)

def pathForSavingModels(model, **params):
    base_dir = params.get("base_dir")
    product_and_product_type = params.get("product_and_product_type")
    predict_type = params.get("predict_type")
    step = params.get("step")
    period = params.get("period")
    product_attribute = params.get("attribute")
    raw_file_name = params.get("raw")
    current_work_path = os.getcwd()
    save_path = f"{current_work_path}/{base_dir}/{predict_type}/{period}_lead_{step}/{product_and_product_type}/{product_attribute}/{raw_file_name}/{model}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path

def retriveBestModelPath(model_path):
    default = "/lightning_logs/version_0/checkpoints"
    try:
        length = len(os.listdir(model_path + default))
    except FileNotFoundError:
        return False

    if length != 0:
        model_path = model_path + f"{default}/" + os.listdir(model_path + default)[0]
        return model_path
    
    return False