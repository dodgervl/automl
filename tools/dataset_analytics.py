import pandas as pd


def missing_and_unique_data(data):
    temp = pd.DataFrame()
    temp['rows_with_null'] = data.isnull().sum()
    temp['percent_of_null'] = round(temp['rows_with_null']/len(data)*100, 1)
    temp['number_of_unique'] = data.nunique()
    temp['percent_of_unique'] = round(temp['number_of_unique']/len(data)*100, 1)
    temp['type'] = data.dtypes
    return temp


def calculate_correlation(train, target_name):
    return train.drop(target_name, axis=1).corr()

