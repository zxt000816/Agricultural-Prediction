import torch
from math import floor
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression

from AnalyzeTools.models import autoregressive_integrated_moving_average, linear_regression, support_vector_regression, random_forest, gradient_boosting
from AnalyzeTools.prepare import data_split, model_eval, pathForSavingModels
from AnalyzeTools.preprocess import preprocessData
from AnalyzeTools.superModels import DEEPAR, TFT, RNN

''' Read data '''
params_path = '../Parameters'
raw_file_name = '경략가격집계 - 소,돼지'
product = "pork"
product_attribute = "경락가격"

_output = 'MAX_COST_AMT' # MIN_COST_AMT
default_exclude_cols = ['JUDGE_GUBN', 'JUDGE_BREED', 'JUDGE_SEX', 'SABLE_GUBN', 'ABATT_CODE']

df = pd.read_csv('../Data/beef/경략가격집계 - 소,돼지.csv', encoding = 'euc_kr', engine ='python').query("JUDGE_KIND == 2")

df = df.drop(default_exclude_cols, axis=1)
df = df.groupby(['STD_DATE']).mean().reset_index()
df['STD_DATE'] = df['STD_DATE'].apply(lambda x: "20" + "-".join(x.split("/")))

df, _input = preprocessData(df, 'STD_DATE', _output)

# prepare dataset for ML or DL
ml_split_params = {'Model': 'ML', 'Future': 1}
X_train, X_test, y_train, y_test = data_split(df, input_cols=_input, output=_output, train_size=0.8, **ml_split_params)

''' Input data into models and Evaluate model results '''
ml_searchCV_params = {
    'base_dir': params_path,
    'product': product,
    'attribute': product_attribute,
    'raw': raw_file_name,
    'save': True
}
stdout = True
vis = False

print("\nARIMA")
arima = autoregressive_integrated_moving_average(y_train)
model_eval(y_test, arima.predict(n_periods=len(y_test), return_conf_int=False, aplha=0.05), stdout=stdout, vis=vis)

print("\nLinear Regression")
lr, _ = linear_regression(X_train, y_train)
model_eval(y_test, lr.predict(X_test), stdout=stdout, vis=vis)

print("\nSupport Vector Regression")
svr, _ = support_vector_regression(X_train, y_train, search=True, **ml_searchCV_params)
model_eval(y_test, svr.predict(X_test), stdout=stdout, vis=vis)

print("\nRandom Forest")
rf, _ = random_forest(X_train, y_train, search=True, **ml_searchCV_params)
model_eval(y_test, rf.predict(X_test), stdout=stdout, vis=vis)

print("\nGradient Boosting")
gb, _ = gradient_boosting(X_train, y_train, search=True, **ml_searchCV_params)
model_eval(y_test, gb.predict(X_test), stdout=stdout, vis=vis)

data = df.copy()

data['time_idx'] = range(len(data))
data['group'] = product

training_cutoff = floor(len(data) * 0.8)

max_prediction_length = 1
max_encoder_length = 30 # 7, 14, 30, 60, 120
batch_size = 64

group = ['group']
time_varying_known_categoricals = ['month', 'week']
time_varying_unknown_categoricals = []
time_varying_known_reals = ['time_idx']
time_varying_unknown_reals = _input + [_output]

print("\nLSTM")
lstm, val_dataloader = RNN(
    data,
    training_cutoff,
    _output,
    group,
    max_encoder_length,
    max_prediction_length,
    time_varying_known_categoricals,
    time_varying_unknown_categoricals,
    time_varying_known_reals,
    batch_size,
    pathForSavingModels(product, product_attribute, raw_file_name, 'LSTM'),
    'LSTM'
)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = lstm.predict(val_dataloader)

model_eval(actuals, predictions, stdout=True, vis=True)

print("\nGRU")
gru, val_dataloader = RNN(
    data,
    training_cutoff,
    _output,
    group,
    max_encoder_length,
    max_prediction_length,
    time_varying_known_categoricals,
    time_varying_unknown_categoricals,
    time_varying_known_reals,
    batch_size,
    pathForSavingModels(product, product_attribute, raw_file_name, 'GRU'),
    'GRU'
)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = gru.predict(val_dataloader)

model_eval(actuals, predictions, stdout=True, vis=True)

print("\nDeepAR")
deep_ar, val_dataloader = DEEPAR(
    data,
    training_cutoff,
    _output,
    group,
    max_encoder_length,
    max_prediction_length,
    time_varying_known_categoricals,
    time_varying_unknown_categoricals,
    time_varying_known_reals,
    batch_size,
    pathForSavingModels(product, product_attribute, raw_file_name, 'DEEPAR'),
)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = deep_ar.predict(val_dataloader)

model_eval(actuals, predictions, stdout=True, vis=True)

print("\nTFT")
tft, val_dataloader = TFT(
    data,
    training_cutoff,
    _output,
    group,
    max_encoder_length,
    max_prediction_length,
    time_varying_unknown_categoricals,
    time_varying_known_categoricals,
    time_varying_known_reals,
    time_varying_unknown_reals,
    batch_size,
    pathForSavingModels(product, product_attribute, raw_file_name, 'TFT'),
)

actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = tft.predict(val_dataloader)

model_eval(actuals, predictions, stdout=True, vis=True)