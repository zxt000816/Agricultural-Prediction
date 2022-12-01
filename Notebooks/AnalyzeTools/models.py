import torch, os, optuna
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from .prepare import read_best_params, save_best_params
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

def file_saver(base_dir, product, attribute, model_name, raw, predict_type, period, step):
    file_name = f"Best_{model_name}.json"

    root_dir = os.getcwd()
    save_dir = f"{root_dir}/{base_dir}/{predict_type}/{period}_lead_{step}/{product}/{attribute}/{raw}/{model_name}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return f"{save_dir}/{file_name}"

def params_extractor(_dict):
    base_dir = _dict.get('base_dir')
    product = _dict.get('product_and_product_type')
    attribute = _dict.get('attribute')
    raw = _dict.get('raw')
    predict_type = _dict.get('predict_type')
    period = _dict.get('period')
    step = _dict.get('step')
    return base_dir, product, attribute, raw, predict_type, period, step

# def best_ml(X, y, model, params, base_dir, product, attribute, model_name, raw, save=True):
#     file_path = file_saver(base_dir, product, attribute, model_name, raw)
#     file_existed = os.path.exists(file_path)

#     if file_existed:
#         params = read_best_params(file_path, model_type='NDL')
    
#     model = GridSearchCV(estimator=model, param_grid=params)
#     model.fit(X, y)
    
#     if save and (not file_existed):
#         save_best_params(model.best_params_, file_path)

#     return model, params if file_existed else model.best_params_

def HyperparamsTuning(study, obj, n_trails, patience=3, verbose=True):
    if patience < 0:
        raise ValueError("Patience must be positive number!")
        
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
        verbose = False
    else:
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(obj, n_trials=n_trails)

    try:
        study.best_params
    except ValueError:
        if patience != 0:
            HyperparamsTuning(study, obj, n_trails, patience-1)
        
        raise ValueError("Cannot find best paramters! Please reset parameters or check data!")
    
    return study

def autoregressive_integrated_moving_average(y_train, y_test, step=1):
    SARIMA_model = pm.auto_arima(y_train, error_action='ignore', suppress_warnings=True, stepwise=False, seasonal=False)
    predictions = []

    for i in range(len(y_test)):
        model = ARIMA(np.append(y_train, y_test[:i]), order=SARIMA_model.order).fit()
        predictions.append(model.forecast(step)[0])
    
    return np.array(predictions)

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    return model, None

# Version 1
# def support_vector_regression(X, y, search=False, **params):
#     model = SVR()
#     best_params = None
    
#     if not search:
#         model.fit(X, y)
#         return model, best_params

#     parameters, base_dir, product, attribute, raw, save = params_extractor(params)
#     parameters = parameters if parameters else {
#         'C': [0.1, 1, 100, 1000],
#         'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
#         'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
#     }

#     model, best_params = best_ml(X, y, model, parameters, base_dir, product, attribute, 'SVR', raw, save)
#     return model, best_params

# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

def support_vector_regression(X, y, search=False, save=False, **params):
    base_dir, product, attribute, raw, predict_type, period, step = params_extractor(params)
    if save and (not search):
        raise ValueError("If save is set as True, search must be True!")

    def support_vector_regression_objective(trial, X, y):
        C = trial.suggest_categorical('C', [0.1,1, 10, 100])
        gamma = trial.suggest_categorical('gamma', [1,0.1,0.01,0.001])
        epsilon = trial.suggest_categorical('epsilon', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
        kernel = trial.suggest_categorical('kernel', ['rbf', 'sigmoid'])

        model = SVR(
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            kernel=kernel,
        )

        score = cross_val_score(model, X, y, cv=5, scoring='r2')
        if np.isnan(score.mean()):
            raise optuna.structs.TrialPruned()
        return score.mean()
    
    best_params = None
    if not search:
        model = SVR()
        model.fit(X, y)
        return model, best_params
    
    file_path = file_saver(base_dir, product, attribute, 'SVR', raw, predict_type, period, step)
    file_existed = os.path.exists(file_path)

    if file_existed:
        print("--> Use the existed best parameters!")
        best_params = read_best_params(file_path)
        print(f"\nBest parameter for SVR is:\n  {best_params}")
        model = SVR(
            C=best_params.get('C'),
            gamma=best_params.get('gamma'),
            epsilon=best_params.get('epsilon'),
            kernel=best_params.get('kernel'),
        )
        model.fit(X, y)
        return model, best_params
    
    print("--> Start searching best parameters!")
    obj = lambda trial: support_vector_regression_objective(trial, X, y)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study = HyperparamsTuning(study, obj, 30, verbose=False)
    best_params = study.best_params
    print(f"\nBest parameter for SVR is:\n  {best_params}")
    if save:
        save_best_params(best_params, file_path)

    model = SVR(
        C=best_params.get('C'),
        gamma=best_params.get('gamma'),
        epsilon=best_params.get('epsilon'),
        kernel=best_params.get('kernel'),
    )
    model.fit(X, y)
    return model, best_params

# Version 1
# def random_forest(X, y, search=False, **params):
#     model = RandomForestRegressor(random_state=0)
#     best_params = None

#     if not search:
#         model.fit(X, y)
#         return model, best_params
    
#     parameters, base_dir, product, attribute, raw, save = params_extractor(params)
#     parameters = parameters if parameters else {
#         'n_estimators': [50, 100, 150], 
#         'max_depth': [3, 5, 7, 9],
#         'max_features': [1, 3, 5, 7, 9],
#         'min_samples_leaf': [1, 2, 3],
#         'min_samples_split': [1, 2, 3],
#     }

#     model, best_params = best_ml(X, y, model, parameters, base_dir, product, attribute, 'RF', raw, save)
#     return model, best_params 

def random_forest(X, y, search=False, save=False, **params):
    base_dir, product, attribute, raw, predict_type, period, step = params_extractor(params)
    if save and (not search):
        raise ValueError("If save is set as True, search must be True!")

    def random_forest_objective(trial, X, y):
        n_estimators = trial.suggest_int('n_estimators', low=10, high=300, step=20)
        max_depth = trial.suggest_int('max_depth', low=1, high=10, step=2)
        max_features = trial.suggest_int('max_features', low=1, high=10, step=2)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', low=1, high=10, step=2)
        min_samples_split = trial.suggest_int('min_samples_split', low=1, high=10, step=2)

        model = RandomForestRegressor(
            random_state=123,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )

        score = cross_val_score(model, X, y, cv=5, scoring='r2')
        if np.isnan(score.mean()):
            raise optuna.structs.TrialPruned()
        return score.mean()
    
    best_params = None
    if not search:
        model = RandomForestRegressor(random_state=123)
        model.fit(X, y)
        return model, best_params
    
    file_path = file_saver(base_dir, product, attribute, 'RF', raw, predict_type, period, step)
    file_existed = os.path.exists(file_path)

    if file_existed:
        print("--> Use the existed best parameters!")
        best_params = read_best_params(file_path)
        print(f"\nBest parameter for Random forest is:\n  {best_params}")
        model = RandomForestRegressor(
            random_state=123,
            n_estimators=best_params.get('n_estimators'),
            max_depth=best_params.get('max_depth'),
            max_features=best_params.get('max_features'),
            min_samples_leaf=best_params.get('min_samples_leaf'),
            min_samples_split=best_params.get('min_samples_split'),
        )
        model.fit(X, y)
        return model, best_params
    
    print("--> Start searching best parameters!")
    obj = lambda trial: random_forest_objective(trial, X, y)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study = HyperparamsTuning(study, obj, 30, verbose=False)
    best_params = study.best_params
    print(f"\nBest parameter for Random forest is:\n  {best_params}")
    if save:
        save_best_params(best_params, file_path)

    model = RandomForestRegressor(
        random_state=123,
        n_estimators=best_params.get('n_estimators'),
        max_depth=best_params.get('max_depth'),
        max_features=best_params.get('max_features'),
        min_samples_leaf=best_params.get('min_samples_leaf'),
        min_samples_split=best_params.get('min_samples_split'),
    )
    model.fit(X, y)
    return model, best_params

# Version 1
# def gradient_boosting(X, y, search=False, **params):
#     model = GradientBoostingRegressor(random_state=0)
#     best_params = None

#     if not search:
#         model.fit(X, y)
#         return model, best_params
    
#     parameters, base_dir, product, attribute, raw, save = params_extractor(params)
#     parameters = parameters if parameters else {
#         'n_estimators': [50, 100, 150], 
#         'max_depth': [3, 5, 7, 9],
#         'max_features': [1, 3, 5, 7, 9],
#         'min_samples_leaf': [1, 2, 3],
#         'min_samples_split': [1, 2, 3],
#     }

#     model, best_params = best_ml(X, y, model, parameters, base_dir, product, attribute, 'GB', raw, save)
#     return model, best_params 

def gradient_boosting(X, y, search=False, save=False, **params):
    base_dir, product, attribute, raw, predict_type, period, step = params_extractor(params)
    if save and (not search):
        raise ValueError("If save is set as True, search must be True!")

    def gradient_boosting_objective(trial, X, y):
        n_estimators = trial.suggest_int('n_estimators', low=10, high=300, step=20)
        max_depth = trial.suggest_int('max_depth', low=1, high=10, step=2)
        max_features = trial.suggest_int('max_features', low=1, high=10, step=2)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', low=1, high=10, step=2)
        min_samples_split = trial.suggest_int('min_samples_split', low=1, high=10, step=2)

        model = GradientBoostingRegressor(
            random_state=123,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )

        score = cross_val_score(model, X, y, cv=5, scoring='r2')
        if np.isnan(score.mean()):
            raise optuna.structs.TrialPruned()
        return score.mean()
    
    best_params = None
    if not search:
        model = GradientBoostingRegressor(random_state=123)
        model.fit(X, y)
        return model, best_params
    
    file_path = file_saver(base_dir, product, attribute, 'GB', raw, predict_type, period, step)
    file_existed = os.path.exists(file_path)

    if file_existed:
        print("--> Use the existed best parameters!")
        best_params = read_best_params(file_path)
        print(f"\nBest parameter for Gradient Boosting is:\n  {best_params}")
        model = GradientBoostingRegressor(
            random_state=123,
            n_estimators=best_params.get('n_estimators'),
            max_depth=best_params.get('max_depth'),
            max_features=best_params.get('max_features'),
            min_samples_leaf=best_params.get('min_samples_leaf'),
            min_samples_split=best_params.get('min_samples_split'),
        )
        model.fit(X, y)
        return model, best_params
    
    print("--> Start searching best parameters!")
    obj = lambda trial: gradient_boosting_objective(trial, X, y)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study = HyperparamsTuning(study, obj, 30, verbose=False)
    best_params = study.best_params
    print(f"\nBest parameter for Gradient Boosting is:\n  {best_params}")
    if save:
        save_best_params(best_params, file_path)

    model = GradientBoostingRegressor(
        random_state=123,
        n_estimators=best_params.get('n_estimators'),
        max_depth=best_params.get('max_depth'),
        max_features=best_params.get('max_features'),
        min_samples_leaf=best_params.get('min_samples_leaf'),
        min_samples_split=best_params.get('min_samples_split'),
    )
    model.fit(X, y)
    return model, best_params

# def lstm(X, y, **params):
#     hidden_sz = params.get("hidden_sz") if params.get("hidden_sz") else 64
#     number_layers = params.get("num_layers") if params.get("num_layers") else 1
#     lr = params.get("lr") if params.get("lr") else 0.001
#     num_epochs = params.get("num_epochs") if params.get("num_epochs") else 100
#     device = params.get("device") if params.get("device") else 'cpu'

#     model = LSTM(
#         output_dim=1, 
#         input_dim=X.shape[-1], 
#         hidden_dim=hidden_sz,  
#         num_layers=number_layers
#     ).float().to(device)
#     criterion = torch.nn.MSELoss(reduction="mean").to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(num_epochs):
#         outputs = model(X.float())
#         optimizer.zero_grad()

#         loss = criterion(outputs, y.view(-1,1).float())
#         loss.backward()

#         optimizer.step()
    
#     return model

# def gru(X, y, **params):
#     hidden_sz = params.get("hidden_sz") if params.get("hidden_sz") else 64
#     number_layers = params.get("num_layers") if params.get("num_layers") else 1
#     lr = params.get("lr") if params.get("lr") else 0.001
#     num_epochs = params.get("num_epochs") if params.get("num_epochs") else 100
#     device = params.get("device") if params.get("device") else 'cpu'

#     model = GRU(
#         output_dim=1, 
#         input_dim=X.shape[-1], 
#         hidden_dim=hidden_sz,  
#         num_layers=number_layers
#     ).float().to(device)
#     criterion = torch.nn.MSELoss().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(num_epochs):
#         outputs = model(X.float())
#         optimizer.zero_grad()

#         loss = criterion(outputs, y.view(-1,1).float())
#         loss.backward()

#         optimizer.step()
    
#     return model

# class LSTM(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_dim, output_dim)
#     def forward(self, x, device=None):
#         if not device:
#             device = x.device
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
#         out = self.fc(out[:, -1, :]) 
#         return out

# class GRU(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
#         super(GRU, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = torch.nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, device=None):
#         if not device:
#             device = x.device
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
#         out, (hn) = self.gru(x, (h0.detach()))
#         out = self.fc(out[:, -1, :]) 
#         return out