path_to_datasets = "./datasets/"
path_to_preprocessed_datasets = "./preprocessed/"
save_preprocessed_data = True

params = {
    "dataset_name": "house_prices",
    "task": "regression",
    "target_name": "SalePrice",
    "features_names": [],
    "create_validation_dataset": True,
    "index_name": "id",
}

validation_params = {
    "validation_split_type": "random",  # order if you would like split data as is without shuffle, appropriete for timeseries
    "percent_on_train": 0.8
}