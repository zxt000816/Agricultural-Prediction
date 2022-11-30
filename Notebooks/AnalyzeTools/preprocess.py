import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from termcolor import colored

def preprocessData(dataframe, time_col, target, prepared=False, fs=True, parse_date=True):
    # prerequirments:
    # dataframe must contain time columns that formatted as XXXX-XX-XX or XXXX/XX/XX
    # dataframe must contain target columns, such as retail price
    # if prepared is set as True, skip step 1 ~ 3, and 5
    # if fs (feature selection) set as False, skip step 4

    df = dataframe.copy()
    df[target] = df[target].apply(lambda x: np.nan if x == 0 else x)
    df[target] = df[target].interpolate(method='linear', limit_direction='both')
    
    if not prepared:
        # step 1
        features = removeNoCorrCols(df, target)
        df = df[[time_col] + features + [target]]

        # step 2
        removeFirstNaRows(df, features)

        # step 3
        fillNa(df, features)
    
    # step 4
    if fs:
        features = featureSelection(df, features, target)
    
    # step 5
    if not prepared:
        features = filterSameCols(df, features)
    
    df.index = range(len(df))
    print(f"\n-->Final features:\n  {features}")
    # parse date
    if parse_date:
        dateParser(df, time_col)
        df = df[[time_col, 'year', 'month', 'week', 'day'] + features + [target]]
    else:
        df = df[[time_col] + features + [target]]

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
        dataframe.interpolate(method='linear', limit_direction='forward', inplace=True)

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

def filterSameCols(dataframe, features, shreshold=0.8):
    length = len(features)
    if length < 2:
        print("Too few features to filter!")
        return features

    filter = np.repeat([True], length)
    for i in range(length):
        for j in range(i+1, length):
            col1, col2 = features[i], features[j]
            percentage = (dataframe[col1] == dataframe[col2]).sum() / len(dataframe)
            if percentage > shreshold:
                if j == length - 1:
                    filter[i] = False
                else:
                    filter[j] = False

    return np.array(features)[filter].tolist()

def dateParser(dataframe, time_col):
    dataframe[time_col] = pd.to_datetime(dataframe[time_col])
    dataframe['year'] = dataframe[time_col].dt.year
    dataframe['month'] = dataframe[time_col].dt.month
    dataframe['week'] = dataframe[time_col].dt.isocalendar().week
    dataframe['day'] = dataframe[time_col].dt.day