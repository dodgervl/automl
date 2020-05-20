import pandas as pd
import config
import tools.dataset_analytics as dataset_analytics
import random
import feature_master
import os
import fitter
from sklearn.externals import joblib
import json


def save(train, val):
    save_path = config.path_to_preprocessed_datasets + config.params["dataset_name"] + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train.to_csv(save_path + "train_preprocessed.csv", index=False)
    val.to_csv(save_path + "val_preprocessed.csv", index=False)
    return


def save_model(best_model):
    save_path = config.path_to_preprocessed_datasets + config.params["dataset_name"] + '/models/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if best_model[1] == 'lgbm':
        save_path += "lgbm/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        best_model[0].booster_.save_model(save_path+'lgbm.txt')
    elif best_model[1] == 'rf':
        save_path += "rf/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        joblib.dump(best_model[0], save_path+'rf.pkl')
    else:
        print('Model wasnt saved!')
    with open(save_path+'params.json', 'w') as f:
        json.dump(best_model[2], f)


def create_val(train):
    p = config.validation_params['percent_on_train']  # percent of data to use for validation
    index = int(train.shape[0] * p)
    if config.validation_params['validation_split_type'] == 'random':
        print(f'Split on train, val using RANDOM {100 * p}% rows as train and left out part as val')
        random.seed(43)
        ids = train.index.tolist()
        random.shuffle(ids)
        val = train.iloc[ids[index:]]
        train = train.iloc[ids[:index]]
    elif config.validation_params['validation_split_type'] == 'order':
        val = train[index:]
        train = train[:index]
        print(f'Split on train, val using FIRST {100 * p}% rows as train and left out part as val')
    print(f'train size: {train.shape[0]}, val size: {val.shape[0]}')
    return train, val


def main():
    # get datasets
    target_name = config.params['target_name']
    index_name = config.params['index_name']
    path = config.path_to_datasets + config.params["dataset_name"] + '/'
    print('Get train dataset')
    train = pd.read_csv(path + "train.csv")
    print('Get val dataset')
    if config.params['create_validation_dataset']:
        train, val = create_val(train)
    else:
        try:
            val = pd.read_csv(path+"val.csv")
        except:
            print('No validation set given! Change config "create_validation_dataset" to True value or give validation file')
            return -1
    # choose features
    print("Feature selection")
    if config.params['features_names']:
        print(f'Using user specified columns: {config.params["features_names"]} as features')
        try:
            train = train[config.params['features_names']]
        except:
            print('Specified columns have error in them')
    analysis_dic = {'null_unique_table': dataset_analytics.missing_and_unique_data(train),
                    'corr_table': dataset_analytics.calculate_correlation(train, target_name)}
    if analysis_dic['null_unique_table'].loc[target_name].percent_of_null > 0:
        print(f'Target name have null values. Please fill in target values or drop them!')
        return -1
    train, val = feature_master.choose_features(train, val, analysis_dic, target_name, index_name)
    # modify and create some features
    train, val, feature_columns = feature_master.prepare_data(train, val, analysis_dic, target_name, index_name)
    if train.empty:
        return -1
    # save preprocessed datasets
    if config.save_preprocessed_data:
        print("Save preprocessed train, val datasets")
        save(train, val)
    best_model = fitter.select_model(train[feature_columns], train[target_name], val[feature_columns], val[target_name], config.params["task"])
    if not best_model:
        return -1
    save_model(best_model)
    return best_model


if __name__ == '__main__':
    response = main()
    if response == -1:
        print('AutoML failed!')
    else:
        print(f'AutoML succeeded! Best model is from {response[1]} family.')
        path = config.path_to_preprocessed_datasets + config.params["dataset_name"] + '/models/'+"lgbm/"
        print(f'You can find model and parameters in following folder: {path}')
