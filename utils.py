import os
import numpy as np
from configuration import CONFIG


def replace_if_condition(condition, replace_value_if_true, replace_value_if_false=None):
    def f(x):
        if condition(x):
            return replace_value_if_true
        if replace_value_if_false is None:
            return x
        else:
            return replace_value_if_false
    return f


def add_suffix_to_files(dir_path, suffix):
    for file_name in set(os.listdir(dir_path)):
        os.rename(os.path.join(dir_path, file_name), os.path.join(dir_path, file_name+suffix))


def prepare_sample_dir(sample_dir_path):
    add_suffix_to_files(os.path.join(sample_dir_path, "csv"), ".csv")
    os.mkdir(os.path.join(sample_dir_path, "images"))


def get_train_test_data(cdb, img_nums=None, columns=None, normalize=False, rebalance=True, seed=None):
    if isinstance(img_nums, int):
        img_nums = [img_nums]

    train_centers = cdb.filter(rebalance=rebalance, seed=seed,
                               img_nums=CONFIG.all_samples.difference(img_nums), columns=columns)
    if train_centers is None:
        return None
    data_train = train_centers[CONFIG.relevant_columns]
    if normalize:
        # Center and norm the data (mean=0, variance=1)
        data_train = data_train.apply(lambda s: s - data_train.mean(), axis=1)
        data_train = data_train.apply(lambda s: s/np.sqrt(data_train.var()), axis=1)
    targets_train = train_centers['diagnosis']

    test_centers = cdb.filter(img_nums=img_nums, columns=columns)
    if test_centers is None:
        return None
    data_test = test_centers[CONFIG.relevant_columns]
    if normalize:
        # Center and norm, based on train data
        data_test = data_test.apply(lambda s: s - data_train.mean(), axis=1)
        data_test = data_test.apply(lambda s: s/np.sqrt(data_train.var()), axis=1)
    targets_test = test_centers['diagnosis']

    return data_train, targets_train, data_test, targets_test
