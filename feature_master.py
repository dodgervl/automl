import numpy as np
import config
import pandas as pd
import os


def get_cols_to_remove(analysis_dic):
    # remove features with null values percent >= p
    p = 40
    remove_cols = analysis_dic['null_unique_table'][
        analysis_dic['null_unique_table'].percent_of_null >= p].index.tolist()
    # remove features with values equals to number of rows of dataset (Highly likely it's indices)
    remove_cols.extend(
        analysis_dic['null_unique_table'][analysis_dic['null_unique_table'].percent_of_unique == 100].index.tolist())
    # remove correlated features
    corr_matrix = analysis_dic['corr_table'].abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    th = 0.8
    remove_cols.extend([column for column in upper.columns if any(upper[column] > th)])
    return list(set(remove_cols))


def choose_features(train, val, analysis_dic, target_name, index_name):
    # select features to remove from dataset
    remove_cols = get_cols_to_remove(analysis_dic)
    if index_name in remove_cols:
        remove_cols.remove(index_name)
    train.drop(remove_cols, axis=1, inplace=True)
    val.drop(remove_cols, axis=1, inplace=True)
    # check consistency between train and val
    train_cols, val_cols = list(train.drop(target_name, axis=1).columns), list(val.drop(target_name, axis=1).columns)
    train_cols.sort()
    val_cols.sort()
    assert train_cols == val_cols, 'Features mismatch between train and val'
    print(f'Features left after removal: {train_cols}')
    return train, val


def handle_missing(train, val):
    train = train.dropna()
    val = val.dropna()
    return train, val


def apply_encoding(train, encoder, feature_name):
    idxes = train.index
    train = pd.merge(train, encoder, how='left', left_on=feature_name, right_on=feature_name)
    train = train.fillna(0)
    train.index = idxes
    return train


def get_encoder(encoder_func, train, target_name, c, save_path):
    path = save_path + c + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    if encoder_func == 'mean':
        encoder = train.groupby(c)[target_name].mean().reset_index()
        encoded_feature = f"{c}_{encoder_func}"
        encoder.columns = [c, encoded_feature]
        encoder.to_csv(path + f"{encoder_func}_encoder_{c}.csv", index=False)
    elif encoder_func == 'var':
        encoder = train.groupby(c)[target_name].var().reset_index()
        encoded_feature = f"{c}_{encoder_func}"
        encoder.columns = [c, encoded_feature]
        encoder.to_csv(path + f"{encoder_func}_encoder_{c}.csv", index=False)
    return encoder, encoded_feature


def encode_categorical(train, val, categorical_features, target_name, feature_columns):
    save_path = config.path_to_preprocessed_datasets + config.params["dataset_name"] + '/' + "encoders/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    encoder_funcs = ['mean', 'var']
    for c in categorical_features:
        encoders = []
        feature_columns.remove(c)
        for encoder_func in encoder_funcs:
            encoder, encoded_feature = get_encoder(encoder_func, train, target_name, c, save_path)
            feature_columns.append(encoded_feature)
            encoders.append(encoder)
        for enc in encoders:
            train = apply_encoding(train, enc, c)
            val = apply_encoding(val, enc, c)
        del train[c]
        del val[c]
    print(f"Features after encoding: {feature_columns}")
    return train, val, feature_columns


def prepare_data(train, val, analysis_dic, target_name, index_name):
    feature_columns = list(train.columns)
    # remove target and index from features to use for modeling
    feature_columns.remove(target_name)
    if index_name in feature_columns:
        feature_columns.remove(index_name)
    null_unique_table = analysis_dic['null_unique_table'].loc[feature_columns]
    # deal with missing data
    train, val = handle_missing(train, val)
    # get categorical features
    categorical_features = null_unique_table[null_unique_table['percent_of_unique'] < 10].index.tolist()
    if config.params['task'] == 'regression':
        # encode categorical features
        train, val, feature_columns = encode_categorical(train, val, categorical_features, target_name, feature_columns)
    else:
        print('This type of task is not yet supported by our AutoML.')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return train, val, feature_columns
