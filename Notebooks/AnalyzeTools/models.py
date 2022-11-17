import torch, os
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from .prepare import read_best_params, save_best_params
import pmdarima as pm

def file_name_formator(product, attribute, model_name, raw):
    return f"{product}_{attribute}_{model_name}__({raw}).json"

def file_path_formator(base_dir, file_name):
    return f"{base_dir}/{file_name}"

def params_extractor(_dict):
    parameters = _dict.get('parameters')
    base_dir = _dict.get('base_dir')
    product = _dict.get('product')
    attribute = _dict.get('attribute')
    raw = _dict.get('raw')
    save = _dict.get('save')
    return parameters, base_dir, product, attribute, raw, save

def best_ml(X, y, model, params, base_dir, product, attribute, model_name, raw, save=True):
    file_name = file_name_formator(product, attribute, model_name, raw)
    file_path = file_path_formator(base_dir, file_name)
    file_existed = os.path.exists(file_path)

    if file_existed:
        params = read_best_params(file_path, model_type='NDL')
    
    model = GridSearchCV(estimator=model, param_grid=params)
    model.fit(X, y)
    
    if save and (not file_existed):
        save_best_params(model.best_params_, file_path)

    return model, params if file_existed else model.best_params_

def autoregressive_integrated_moving_average(y_train):
    model = pm.auto_arima(y_train, error_action='ignore', suppress_warnings=True, stepwise=False, seasonal=False)
    return model

def linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    return model, None

def support_vector_regression(X, y, search=False, **params):
    model = SVR()
    best_params = None
    
    if not search:
        model.fit(X, y)
        return model, best_params

    parameters, base_dir, product, attribute, raw, save = params_extractor(params)
    parameters = parameters if parameters else {
        'C': [1, 10]
    }

    model, best_params = best_ml(X, y, model, parameters, base_dir, product, attribute, 'SVR', raw, save)
    return model, best_params
    
def random_forest(X, y, search=False, **params):
    model = RandomForestRegressor(random_state=0)
    best_params = None

    if not search:
        model.fit(X, y)
        return model, best_params
    
    parameters, base_dir, product, attribute, raw, save = params_extractor(params)
    parameters = parameters if parameters else {
        'n_estimators': [100, 150], 
        'max_depth': [1, 2, 3, 4]
    }

    model, best_params = best_ml(X, y, model, parameters, base_dir, product, attribute, 'RF', raw, save)
    return model, best_params 

def gradient_boosting(X, y, search=False, **params):
    model = GradientBoostingRegressor(random_state=0)
    best_params = None

    if not search:
        model.fit(X, y)
        return model, best_params
    
    parameters, base_dir, product, attribute, raw, save = params_extractor(params)
    parameters = parameters if parameters else {
        'n_estimators': [100, 150], 
        'max_depth': [1, 2, 3, 4]
    }

    model, best_params = best_ml(X, y, model, parameters, base_dir, product, attribute, 'GB', raw, save)
    return model, best_params 

def arima(X, y, **params):
    pass

def transformer(X, y, **params):
    pass

def cnn():
    pass

def lstm(X, y, **params):
    hidden_sz = params.get("hidden_sz") if params.get("hidden_sz") else 64
    number_layers = params.get("num_layers") if params.get("num_layers") else 1
    lr = params.get("lr") if params.get("lr") else 0.001
    num_epochs = params.get("num_epochs") if params.get("num_epochs") else 100
    device = params.get("device") if params.get("device") else 'cpu'

    model = LSTM(
        output_dim=1, 
        input_dim=X.shape[-1], 
        hidden_dim=hidden_sz,  
        num_layers=number_layers
    ).float().to(device)
    criterion = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        outputs = model(X.float())
        optimizer.zero_grad()

        loss = criterion(outputs, y.view(-1,1).float())
        loss.backward()

        optimizer.step()
    
    return model

def gru(X, y, **params):
    hidden_sz = params.get("hidden_sz") if params.get("hidden_sz") else 64
    number_layers = params.get("num_layers") if params.get("num_layers") else 1
    lr = params.get("lr") if params.get("lr") else 0.001
    num_epochs = params.get("num_epochs") if params.get("num_epochs") else 100
    device = params.get("device") if params.get("device") else 'cpu'

    model = GRU(
        output_dim=1, 
        input_dim=X.shape[-1], 
        hidden_dim=hidden_sz,  
        num_layers=number_layers
    ).float().to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        outputs = model(X.float())
        optimizer.zero_grad()

        loss = criterion(outputs, y.view(-1,1).float())
        loss.backward()

        optimizer.step()
    
    return model

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x, device=None):
        if not device:
            device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class GRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, device=None):
        if not device:
            device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out