import os
import logging
import numpy as np
import pandas as pd
import math
from scipy.special import erf
from .configuration import CONFIG
logger = logging.getLogger(__name__)
logger.setLevel(CONFIG.logging_level)


class EmptyDataSet(Exception):
    pass


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


def _concatenate_raw_data():
    pix_tup_df = pd.read_csv(CONFIG.raw_pixtup_path, header=None)
    pix_tup_df = pix_tup_df.iloc[:, 0:17]
    xy_df = pd.read_csv(CONFIG.raw_xy_path, header=None)
    npos_df = pd.read_csv(CONFIG.raw_npos_path, header=None)
    concat_df = pd.concat([npos_df, xy_df, pix_tup_df], axis=1)
    concat_df.columns = CONFIG.all_columns
    concat_df.to_csv(CONFIG.current_data_path, index=False)
    return concat_df


def load_df(reload=False):
    """
    Concatenates raw data into one Dataframe. Stores it in a temp file for more efficiency.
    """
    if reload or not os.path.isfile(CONFIG.current_data_path):
        concat_df = _concatenate_raw_data()
    else:
        concat_df = pd.read_csv(CONFIG.current_data_path)
    return concat_df


def get_train_test_data(cdb, test_imgs, train_imgs=None, columns=None, normalize=None, rebalance=None, seed=None):
    # Set defaults
    norm = minkowski(2)
    if isinstance(test_imgs, int):
        test_imgs = [test_imgs]
    if normalize is None:
        normalize = False
    if rebalance is None:
        rebalance = True
    if columns is not None:
        CONFIG.learning_Mij = list(columns)

    data_train = cdb.filter(rebalance=rebalance,
                            seed=seed,
                            img_nums=train_imgs,
                            columns=CONFIG.relevant_columns)
    if data_train is None:
        logger.error("Empty data train: data cannot be splitted. "
                     "This can be the case if you asked for balanced data, but all the data has the same diagnosis.")
        raise EmptyDataSet('Empty train data set')
    targets_train = data_train['diagnosis']
    data_train = data_train[CONFIG.learning_Mij]
    if normalize:
        # Center and norm the data (mean=0, variance=1)
        data_train = (data_train-CONFIG.mean)/CONFIG.std
        data_train = norm(data_train.transpose())
        data_train = data_train.reshape(-1, 1)

    data_test = cdb.filter(img_nums=test_imgs, columns=CONFIG.relevant_columns)
    if data_test is None:
        logger.error("Empty data test: data cannot be splitted.")
        raise EmptyDataSet('Empty test data set')
    targets_test = data_test['diagnosis']
    data_test = data_test[CONFIG.learning_Mij]
    if normalize:
        # Center and norm, based on train data
        data_test = (data_test-CONFIG.mean)/CONFIG.std
        data_test = norm(data_test.transpose())
        data_test = data_test.reshape(-1, 1)

    return data_train, targets_train, data_test, targets_test


def minkowski(p):
    if p == 1:
        def norm(df):
            return abs(df).max()
    elif p == 2:
        def norm(df):
            return np.sqrt(np.square(df).sum(axis=0))
    else:
        raise NotImplementedError(p)
    return norm


def nested(func):
    def wrapper(l):
        if hasattr(l, '__iter__'):
            return [func(x) for x in l]
        # if isinstance(l, pd.Series):
        #     return pd.Series([func(x) for x in l])
        return func(l)
    return wrapper


def normal_cdf(mu, sigma):
    @nested
    def cdf(z):
        return 0.5 * (1 + erf((z-mu)/(sigma*math.sqrt(2))))
    return cdf


def normal_pdf(mu, sigma, alpha=1.0):
    @nested
    def pdf(z):
        return alpha/(sigma*math.sqrt(2*math.pi) * math.exp((z-mu)**2/(2 * sigma**2)))
    return pdf


def _correlation(l1, l2, n):
    return sum([l1[i]*l2[i] for i in range(n)])


def ncc(l1, l2):
    n = len(l1)
    assert(n == len(l2))
    return _correlation(l1, l2, n)/math.sqrt(_correlation(l1, l1, n) * _correlation(l2, l2, n))
