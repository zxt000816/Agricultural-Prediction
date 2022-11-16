import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt
import json, os, torch
import numpy as np

def model_eval(Y_true, Y_predict, stdout=False, vis=False, ret=False, **params):
    if not params:
        MAPE, RSQUARE=  mape(Y_true, Y_predict), R2(Y_true, Y_predict)
    else:
        scaler = params['price_scaler']
        Y_true = scaler.inverse_transform(Y_true.reshape(-1, 1))
        Y_predict = scaler.inverse_transform(Y_predict.reshape(-1, 1))
        MAPE, RSQUARE = mape(Y_true, Y_predict), R2(Y_true, Y_predict)
    if stdout:
        print(f"MAPE: {MAPE} R square: {RSQUARE}")
    if vis:
        vis_result(Y_true, Y_predict)
    if ret:
        return MAPE, RSQUARE

def vis_result(Y_true, Y_predict):
    plt.figure(figsize=(10, 6))
    plt.plot(Y_true, label='Actual')
    plt.plot(Y_predict, label='Predict')
    plt.show()

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
    
def read_best_params(file_path, model_type='NDL'):
    with open(file_path) as f:
        model_best_params = json.load(f)
        if model_type == 'DL':
            return model_best_params
        for key, val in model_best_params.items():
            model_best_params[key] = [val]
        return model_best_params

def data_split(data, input_cols=None, output=None, train_size=None, **params):
    if params.get('Model') == 'ML':
        target = f'Future_{output}'
        data[target] = data[output].shift(-1 * params.get('Future'))
        data = data.dropna()

        input_cols.append(output)

        X_train, X_test, y_train, y_test = train_test_split(
            data[input_cols].values, 
            data[target].values, 
            random_state=0, 
            test_size=1-train_size,
            shuffle=False
        )
        print(f"X_train: {X_train.shape} y_train: {y_train.shape} X_test: {X_test.shape} y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    X, y = make_dataset(data[input_cols].values, data[output].values, params['future'], params['past'])

    X_train, y_train = X[:params['dividing_line'], :, [-1]], y[:params['dividing_line'], -1]
    X_test, y_test = X[params['dividing_line']:, :, [-1]], y[params['dividing_line']:, -1]
    print(f"X_train_dl: {X_train.shape} y_train_dl: {y_train.shape} X_test_dl: {X_test.shape} y_test_dl: {y_test.shape}")

    X_train, y_train = torch.from_numpy(X_train).to(params['device']), torch.from_numpy(y_train).to(params['device'])
    X_test, y_test= torch.from_numpy(X_test).to(params['device']), torch.from_numpy(y_test).to(params['device'])

    return X_train, X_test, y_train, y_test

def make_dataset(X_arr, y_arr, future, past):
    X, y = [], []
    for i in range(past, len(X_arr)-future):
        X.append(X_arr[i-past: i, :])
        y.append(y_arr[i+future-1, [-1]])
    return np.array(X, dtype=np.float16), np.array(y, dtype=np.float16)

def filterSameColumns(dataframe, cols, shreshold=0.8):
    length = len(cols)
    if length < 3:
        return cols

    filter = np.array([True for i in range(length)])
    for i in range(length):
        for j in range(i+1, length):
            col1, col2 = cols[i], cols[j]
            percentage = (dataframe[col1] == dataframe[col2]).sum() / len(dataframe)
            if percentage > shreshold:
                if j == length - 1:
                    filter[i] = False
                else:
                    filter[j] = False

    return np.array(cols)[filter].tolist()[:-1]

