import os
import numpy as np
import pandas as pd
import functools
import logging

from matplotlib import pyplot as plt
from PIL import Image

from ..configuration import CONFIG
from ..utils import minkowski, EmptyDataSet
from learning import neural_networks

logger = logging.getLogger(__name__)
logger.setLevel(CONFIG.logging_level)


def filter_df(df, rebalance=False, seed=None, *, img_nums=None, diagnosis=None, columns=None):
    """
    Selects matching points in the database.

    Parameters
    ==========

    df: pd.Dataframe, the points to be filtered
    img_nums: int or array, images to keep
    diagnosis: int or array, diagnosis to keep
    columns: str or array, columns to keep
    rebalance: bool, whether the result should have as many points of each diagnosis
    seed: int or None, seed of the random generator

    Returns
    =======
    df: The pd.Dataframe of the points matching the filter
    """
    if isinstance(img_nums, int):
        img_nums = [img_nums]
    if isinstance(diagnosis, int):
        diagnosis = [diagnosis]
    if isinstance(columns, str):
        columns = [columns]

    if img_nums is not None:
        df = df[df['img_num'].isin(img_nums)]
    if diagnosis is not None:
        df = df[df['diagnosis'].isin(diagnosis)]
    if rebalance:
        np.random.seed(seed)
        points3 = df[df['diagnosis'] == 3]
        points4 = df[df['diagnosis'] == 4]
        if len(points3) == 0 or len(points4) == 0:
            # Cannot rebalance
            return None
        if len(points3) < len(points4):
            idx = list(points4.index)
            np.random.shuffle(idx)
            df.drop(idx[len(points3):], inplace=True)
        else:
            idx = list(points3.index)
            np.random.shuffle(idx)
            df.drop(idx[len(points4):], inplace=True)
    if columns is not None:
        df = df[df.columns.intersection(columns)]
    return df


class NeuralNetwork:
    def __init__(self, columns=None, layers=None, regularization_lambda=None, pgtol=None, maxfun=None, result_args=(),
                 *, model_name=None):
        self.theta = None
        self.layers = layers
        self.df = pd.read_csv(CONFIG.current_data_path)
        if model_name is None:
            model_name = 'neural_networks'
        else:
            theta_path = os.path.join(CONFIG.model_path, model_name, 'theta')
            self._load_theta(theta_path)
        self.out_path = os.path.join(CONFIG.model_path, model_name)
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        if self.theta is not None:
            return  # Model already loaded

        if layers is None:
            raise AttributeError('You must set either `layers` or `model_name`, to define the layers of the NN.')
        if columns is None:
            raise AttributeError('You must set either `columns` or `model_name`, to define the columns of the NN.')

        # Create useful features
        assert layers[0] == len(columns)
        if isinstance(columns, str):
            columns = [columns]
        assert 'diagnosis' not in columns
        self.columns = columns
        self._create_features()

        # Neural nets parameters
        self.regularization_lambda = regularization_lambda
        self.pgtol = pgtol
        self.maxfun = maxfun
        self.result_args = result_args

    def run_cross_validation(self, k_fold=None, test_imgs=None, train_imgs=None, save_model=True,
                             model_name=None, normalize=None, rebalance=None, seed=None):
        if test_imgs is not None:
            if isinstance(test_imgs, int):
                test_imgs = [test_imgs]
            if train_imgs is None:
                train_imgs = CONFIG.all_samples.difference(test_imgs)
        elif train_imgs is not None:
            if isinstance(train_imgs, int):
                train_imgs = [train_imgs]
            test_imgs = CONFIG.all_samples.difference(train_imgs)
        else:
            if k_fold is None:
                k_fold = 4
            if k_fold <= 1 or not isinstance(k_fold, int):
                raise AttributeError('k_fold should be an integer greater than 2.')
            np.random.seed(seed)
            samples = list(CONFIG.all_samples)
            np.random.shuffle(samples)
            scores = []
            for k in range(k_fold):
                size = int(len(samples)/k_fold)
                test_imgs = samples[k*size:(k+1)*size]
                logger.debug('Running cross-validation %i/%i, testing on %s' % ((k+1), k_fold, str(sorted(test_imgs))))
                try:
                    score = self.run_cross_validation(test_imgs=test_imgs, save_model=(save_model and k == 0),
                                                      model_name=model_name, normalize=normalize, rebalance=rebalance,
                                                      seed=seed)
                except EmptyDataSet:
                    continue
                scores.append(score)

            return scores

        return neural_networks.run(functools.partial(self.get_data, self.df, train_imgs, test_imgs, self.columns.copy(),
                                                     10000, rebalance, seed, normalize),
                                   self.columns, self.layers[-1], self.layers[1:-1],
                                   regularization_lambda=self.regularization_lambda,
                                   out_path=self.out_path, pgtol=self.pgtol, maxfun=self.maxfun,
                                   result_args=self.result_args,
                                   theta=self.theta, layers=self.layers)

    def predict(self, imgs):
        df = filter_df(self.df, img_nums=imgs, columns=self.columns)
        return neural_networks.predict(self.theta, df)

    def predict_and_show(self, img_num, complete_df, width, height):
        assert width * height == len(complete_df)
        logger.info("Predicting neural network results on image %i ..." % img_num)
        img_out_path = os.path.join(self.out_path, "prediction_image%i.jpg" % img_num)
        array = np.zeros((width, height, 3), 'uint8')
        complete_df = self._create_features(complete_df)
        complete_df = complete_df[self.columns]
        prediction = neural_networks.predict(self.theta, complete_df)
        prediction = prediction.reshape((width, height))
        array[..., 0] = 255*(1-prediction)
        array[..., 1] = 255*prediction
        img_num = Image.fromarray(array)
        img_num.save(img_out_path)

    def _create_features(self, df=None):
        if df is None:
            df = self.df
            inplace = True
        else:
            inplace = False
        p = 2
        norm = minkowski(p)
        learning_df = self.df[CONFIG.learning_Mij]
        # Norm
        if 'norm' in self.columns and 'norm' not in df.columns:
            norm_df = (learning_df - learning_df.mean()) / learning_df.std()
            norm_df = norm(norm_df.transpose())
            df['norm'] = norm_df
        # Extrema
        if 'max' in self.columns and 'max' not in df.columns:
            df['max'] = learning_df.transpose().max().transpose()
        if 'min' in self.columns and 'min' not in df.columns:
            df['min'] = learning_df.transpose().min().transpose()
        if 'mean' in self.columns and 'mean' not in df.columns:
            df['mean'] = learning_df.transpose().mean().transpose()
        if 'std' in self.columns and 'std' not in df.columns:
            df['std'] = learning_df.transpose().std().transpose()
        if inplace:
            self.df = df
        return df

    def _load_theta(self, dir_path):
        theta = []
        self.layers = None
        self.columns = None
        for file_name in sorted(os.listdir(dir_path)):
            if file_name.startswith("theta"):
                theta.append(np.array(pd.read_csv(os.path.join(dir_path, file_name), sep=";", header=None)))
            if file_name.startswith("layers"):
                with open(os.path.join(dir_path, file_name)) as in_f:
                    self.layers = [int(x) for x in in_f.readline().split(';')]
            if file_name.startswith("columns"):
                with open(os.path.join(dir_path, file_name)) as in_f:
                    self.columns = in_f.readline().split(';')
                self._create_features()
        self.theta = theta
        if self.layers is None:
            logger.warning("Layers couldn't be retrieved from the model.")
        if self.columns is None:
            logger.warning("Columns couldn't be retrieved from the model.")

    def show_theta_influence(self, out_path=None):
        weights = np.ones((self.layers[-1],))
        theta = self.theta.copy()
        while theta:
            last_theta = theta.pop()
            last_theta = np.delete(last_theta, 0, 1)
            last_theta = np.abs(last_theta)
            weights = np.dot(weights, last_theta)

        weights /= np.max(weights)
        idx = weights.argsort()[::-1]
        self.columns = np.array(self.columns)
        self.columns = self.columns[idx]
        weights = weights[idx]
        fig, ax = plt.subplots()
        plt.plot(weights)
        ax.set_xticks(range(len(self.columns)))
        ax.set_xticklabels(self.columns)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if out_path is not None:
            plt.savefig(out_path)
        plt.show()
        plt.clf()
        return weights

    def get_data(self, df, training_images=None, testing_images=None, learning_columns=None, max_samples=10000,
                 rebalance=None, seed=None, normalize=True):
        if training_images is None:
            training_images = CONFIG.all_samples
        if testing_images is None:
            testing_images = CONFIG.all_samples
        if learning_columns is None:
            learning_columns = CONFIG.learning_Mij
        assert 'diagnosis' not in learning_columns
        all_columns = learning_columns + ['diagnosis']
        training_data = filter_df(df, rebalance=rebalance, seed=seed, img_nums=training_images,
                                  columns=all_columns)

        # Select data
        total_training = len(training_data)
        if max_samples is None or max_samples > total_training:
            max_samples = total_training
        random_training_instances = np.array(training_data.index)
        np.random.seed(seed)
        np.random.shuffle(random_training_instances)
        random_training_instances = random_training_instances[:max_samples]
        training_data = training_data[training_data.index.isin(random_training_instances)]

        training_labels = training_data['diagnosis'] - 3
        print("Learning on the columns", learning_columns)
        training_data = training_data[learning_columns]
        testing_data = filter_df(df, rebalance=rebalance, seed=seed, img_nums=testing_images,
                                 columns=all_columns)
        testing_labels = testing_data['diagnosis'] - 3
        testing_data = testing_data[learning_columns]
        if normalize:
            training_data = (training_data - training_data.mean()) / training_data.std()
            testing_data = (testing_data - testing_data.mean()) / testing_data.std()
        if len(training_labels) and len(testing_labels):
            return np.array(training_data), np.array(training_labels, dtype=np.uint8),\
                   np.array(testing_data), np.array(testing_labels, dtype=np.uint8)
        raise EmptyDataSet()
