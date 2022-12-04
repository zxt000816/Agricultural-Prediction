import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from math import floor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from termcolor import colored
from dateutil.relativedelta import relativedelta

# Currently, only single time series preprocessing is supported
def preprocessData(dataframe, time_col, target, period='Day', prepared=False, fs=True, parse_date=True, fill_missing_dates=True):
    # prerequirments:
    # dataframe must contain time columns that formatted as XXXX-XX-XX or XXXX/XX/XX
    # dataframe must contain target columns, such as retail price
    # if prepared is set as True, skip step 1 ~ 3, and 5
    # if fs (feature selection) set as False, skip step 4

    df = dataframe.copy()

    if not prepared:
        # step 1
        features = removeNoCorrCols(df, target)
        df = df[[time_col] + features + [target]]

        # step 2
        removeFirstNaRows(df, features)

        # step 3:
        fillNa(df, features)
    
    # Resolve error caused by prediction target containing 0
    df[target] = df[target].apply(lambda x: np.nan if x == 0 else x)
    df[target] = df[target].interpolate(method='linear', limit_direction='forward')

    # step 4: select importance features
    if fs:
        features = featureSelection(df, features, target)
    
    # step 5: filter almost same columns
    if not prepared:
        features = filterSameCols(df, features, target)
    
    print(f"\n-->Final features:\n  {features}")

    # step 6: fill missing dates
    if fill_missing_dates:
        df[time_col] = pd.to_datetime(df[time_col])
        
        no_time_cols = [*features, target]
        missing_dates = fillMissingDates(df, time_col, no_time_cols, period)
        
        df = pd.concat((df, missing_dates), axis=0).sort_values([time_col])
        df[no_time_cols] = df[no_time_cols].interpolate(method='linear', limit_direction='forward')

    # parse date
    if parse_date:
        dateParser(df, time_col)
        df = df[[time_col, 'year', 'month', 'week', 'day'] + features + [target]]
    else:
        df = df[[time_col] + features + [target]]

    df.index = range(len(df))
    return df, features

def removeNoCorrCols(dataframe, target):
    features = dataframe.corr()[target].dropna().index.tolist()
    if not features:
        raise ValueError("No features correlated with the target!")
    features.remove(target)

    return features

def removeFirstNaRows(dataframe, features):
    drop_rows = []
    if dataframe[features].isnull().values.any():
        for i, row in dataframe[features].iterrows():
            if row.isnull().values.any():
                drop_rows.append(i)
                continue
            break
    
    dataframe.drop(drop_rows, axis=0, inplace=True)

def fillNa(dataframe, features):
    if dataframe[features].isnull().values.any():
        dataframe[features] = dataframe[features].interpolate(method='linear', limit_direction='forward')

def featureSelection(dataframe, features, target, K=None):
    # use sklearn selectbest function
    if not K:
        if len(features) > 2:
            K = int(len(features) / 2)
        else:
            print(colored("There are too few features in the data. The raw data features will be used.", 'yellow'))
            return features
    
    feature_selector = SelectKBest(score_func=f_regression, k=K)
    feature_selector.fit_transform(dataframe[features].values, dataframe[target].values)

    feature_scores = [[k, v] for k, v in zip(features, feature_selector.scores_)]
    print("\n-->Feature scores:\n  ", pd.DataFrame(feature_scores, columns=['Features', 'Scores']).sort_values('Scores', ascending=False))

    k_best_features = list(np.array(features)[feature_selector.get_support()])
    print("\n-->TOP K features:\n  ", k_best_features)

    return k_best_features

def filterSameCols(dataframe, features, target, shreshold=0.8):
    length = len(features)
    if length < 2:
        print("Too few features to filter!")
        return features

    cols_filter = np.repeat([True], length)
    for i in range(length):
        col1 = features[i]
        output_sim = (dataframe[col1] == dataframe[target]).sum() / len(dataframe)
        if output_sim > shreshold:
            cols_filter[i] = False
            continue

        for j in range(i+1, length):
            col2 = features[j]
            input_sim = (dataframe[col1] == dataframe[col2]).sum() / len(dataframe)
            if input_sim > shreshold:
                cols_filter[j] = False

    return np.array(features)[cols_filter].tolist()

def dateParser(dataframe, time_col):
    dataframe[time_col] = pd.to_datetime(dataframe[time_col])
    dataframe['year'] = dataframe[time_col].dt.year
    dataframe['month'] = dataframe[time_col].dt.month
    dataframe['week'] = dataframe[time_col].dt.isocalendar().week
    dataframe['day'] = dataframe[time_col].dt.day

def fillMissingDates(dataframe, time_col, no_time_cols, period):
    df = dataframe.copy()
    prev_date = None
    missing_dates = []
    for _, row in df.iterrows():
        if prev_date:
            if period == 'Day':
                date_gap = row[time_col].day - prev_date.day
                if date_gap != 1:
                    for d in range(1, date_gap):
                        missing_dates.append([prev_date + relativedelta(days=d), *np.repeat([np.nan], len(no_time_cols))])
            
            if period == 'Week':
                date_gap = row[time_col].week - prev_date.week
                if date_gap != 1:
                    for w in range(1, date_gap):
                        missing_dates.append([prev_date + relativedelta(weeks=w), *np.repeat([np.nan], len(no_time_cols))])

            if period == 'Month':
                date_gap = row[time_col].month - prev_date.month
                if date_gap != 1:
                    for m in range(1, date_gap):
                        missing_dates.append([prev_date + relativedelta(months=m), *np.repeat([np.nan], len(no_time_cols))])

        prev_date = row[time_col]

    missing_dates = pd.DataFrame(missing_dates, columns=[time_col, *no_time_cols])

    return missing_dates

def removeOutliers(dataframe, test_size, target_col, vis=True, **params):
    data = dataframe.copy()

    test_size = floor(len(data) * test_size) if type(test_size) == float else test_size
    training_data = data[:-1*test_size]
    training_idxs = training_data.index

    n_estimators = params.get('n_estimators') if params.get('n_estimators') else 100
    contamination = params.get('contamination') if params.get('contamination') else 0.03
    
    iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination, max_samples='auto')
    prediction = iforest.fit_predict(training_data[[target_col]])

    print("Number of outliers detected: {}".format(prediction[prediction < 0].sum()))
    print("Number of normal samples detected: {}".format(prediction[prediction > 0].sum()))

    if vis:
        normals = []
        outliers = []
        for value, label in zip(training_data[target_col].values, prediction):
            if label not in [1, -1]:
                print(label)
            if label == 1:
                normals.append(value)
                outliers.append(None)
            elif label == -1:
                normals.append(None)
                outliers.append(value)
        plt.figure(figsize=(12,7))
        plt.plot(normals, label='normal')
        plt.plot(outliers, label='outlier')
        plt.legend()
        plt.show()
    
    for idx, label in zip(training_idxs, prediction):
        if label == -1:
            data.loc[idx, target_col] = np.nan

    data[target_col] = data[target_col].interpolate(method='linear', limit_direction='both')

    return data

def createPeriodData(dataframe, operations, period, time_col):
    data = dataframe.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.isocalendar().week
    data['day'] = data['date'].dt.day

    if period == 'Day':
        group_cols = ['date']
    elif period == 'Week':
        group_cols = ['year', 'week']
    elif period == 'Month':
        group_cols = ['year', 'month']
    
    agg_operations = createAggOperations(operations, data.columns, group_cols, ['year', 'month', 'week', 'day'])
    data = data.groupby(group_cols).agg(agg_operations).reset_index()
    data = data.drop(['year', 'month', 'week', 'day'], axis=1)

    return data

def createAggOperations(baseOperations, columns, group_cols, time_cols):
    operations = baseOperations.copy()
    agg_operations = {}
    processed_cols = []

    for k, v in operations.items():
        if k != 'others':
            agg_operations[k] = v
            processed_cols.append(k)
        else:
            for col in columns:
                if col not in processed_cols:
                    if col not in time_cols:
                        agg_operations[col] = operations[k]
                    else:
                        agg_operations[col] = 'first'
        
    for k in agg_operations.copy():
        if k in group_cols:
            agg_operations.pop(k)
            
    return agg_operations