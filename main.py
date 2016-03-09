"""
Main file to run all other ones:
- Save new raw files in database
- Preprocess data
- Train and test models
- Save/Visualize results
"""
import os
import shutil
import logging
import random
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from PIL import Image

from configuration import CONFIG
from utils import get_train_test_data, EmptyDataSet
from database import ClusterDB
from learning import NeuralNetwork


logger = logging.getLogger(__name__)
logger.setLevel(level=CONFIG.logging_level)


def save_new_pixtup_in_database(pixtup_file_name, xy_file_name=None, img_num=None):
    """
    Add a new image to database.
    - You can give one file with the following columns (in this order):
    [x, y, M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44]
    - You can also create two separate files for pixtup and xy, with the following columns (in this order):
    [M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34, M41, M42, M43, M44] and [x, y]

    WARNING
    =======
    The files PixTup.csv, XY.csv and NPOS.csv will be updated.
    Make sure the columns you add match the original ones.

    Parameters
    ==========
    pixtup_file_name: The name of the pixtup file to append to the existing PixTup.csv
    xy_file_name: The name of the new xy file, if the pixtup and xy files are separated.
    img_num: The number of the new image. If None, it will assign the first number not assigned yet.

    Returns
    =======
    Returns True if the new files were successfully added, False otherwise.
    Raises an error if the dimensions of the csv files don't match the existing ones.
    """
    in_path = os.path.join(CONFIG.raw_data_path, pixtup_file_name)
    if not os.path.isfile(in_path):
        logger.error("No file %s was found in raw_data/ folder.")
        raise FileNotFoundError(in_path)
    new_pixtup_df = pd.read_csv(in_path, sep=",", header=None)
    if xy_file_name is None:
        new_xy_df = new_pixtup_df.iloc[:, :2]
        new_pixtup_df = new_pixtup_df.iloc[:, 2:19]
    else:
        new_pixtup_df = new_pixtup_df.iloc[:, 0:17]
        new_xy_path = os.path.join(CONFIG.raw_data_path, xy_file_name)
        if not os.path.isfile(in_path):
            logger.error("No file %s was found in raw_data/ folder.")
            raise FileNotFoundError(new_xy_path)
        new_xy_df = pd.read_csv(new_xy_path, sep=",", header=None)
    old_pixtup_df = pd.read_csv(CONFIG.raw_pixtup_path, header=None).iloc[:, 0:17]
    old_xy_df = pd.read_csv(CONFIG.raw_xy_path, header=None)
    logger.warning("You are about to concatenate the following Dataframes:\n %s \n and \n %s"
                   % (str(old_pixtup_df.head()), str(new_pixtup_df.head())))
    choice = input("Do you want to continue? (y/n)")
    if choice.lower() != 'y':
        logger.warning("Saving new pixtup is cancelled.")
        return False
    logger.warning("You are about to concatenate the following Dataframes:\n %s \n and \n %s"
                   % (str(old_xy_df.head()), str(new_xy_df.head())))
    choice = input("Do you want to continue? (y/n)")
    if choice.lower() != 'y':
        logger.warning("Saving new pixtup is cancelled.")
        return False
    pixtup_df = pd.concat([old_pixtup_df, new_pixtup_df])
    xy_df = pd.concat([old_xy_df, new_xy_df])
    old_npos_df = pd.read_csv(CONFIG.raw_npos_path, header=None)
    if img_num is None:
        imgs = set(old_npos_df)
        img_num = 0
        while img_num in imgs:
            img_num += 1
    npos_df = pd.concat([old_npos_df, [img_num]*len(new_xy_df)])
    if os.path.isdir(os.path.join(CONFIG.backup_data_path, 'raw_data')):
        shutil.rmtree(os.path.join(CONFIG.backup_data_path, 'raw_data'))
        logger.info("Deleted the last backup of raw files.")
    shutil.move(CONFIG.raw_data_path, os.path.join(CONFIG.backup_data_path, 'raw_data'))
    logger.info("Created a backup of raw files.")
    os.mkdir(CONFIG.raw_data_path)
    pixtup_df.to_csv(CONFIG.raw_pixtup_path, index=False, header=None)
    xy_df.to_csv(CONFIG.raw_xy_path, index=False, header=None)
    npos_df.to_csv(CONFIG.raw_npos_path, index=False, header=None)
    logger.info("PixTup, XY and NPOS.csv updated.")
    concat_df = pd.concat([npos_df, xy_df, pixtup_df], axis=1)
    concat_df.columns = CONFIG.all_columns
    concat_df.to_csv(CONFIG.current_data_path, index=False)
    logger.info("Current data updated.")
    return True


def load_estimator(model_path):
    logger.warning("Never unpickle untrusted data! "
                   "You must be sure that %s is the path of a fitted sklearn estimator." % model_path)
    answer = input('Continue? (y/n)')
    if answer.lower() != 'y':
        logger.error('Data was not unpickled.')
        return None
    logger.debug('Unpickling estimator at %s' % model_path)
    return joblib.load(model_path)


def run_cross_validation(estimator, columns, k_fold=None, test_imgs=None, train_imgs=None, save_model=False,
                         model_name=None, normalize=None, rebalance=None, seed=None):
    """
    Run cross-validation for a sklearn estimator, on given columns and images.

    Parameters
    ==========
    estimator: The sklearn estimator or 'neural networks'
    columns: The columns (i.e. Mij) to apply the model on
    k_fold: The number of folds for the cross-validation. Default: 4
    test_imgs: The list of images to test the model on. This is prioritary on k_fold
    train_imgs: The list of images to train the model on. Default if test_imgs is not None: all other images.
    save_model: Whether to save the model or not after it is trained.
    model_name: The name of the model. Default: 'model'
    normalize: Whether to normalize data before applying the model.
    rebalance: Whether to select as many points of each type (healthy or ill).
    seed: The seed of the random generator.

    """
    if isinstance(estimator, NeuralNetwork):
        run_neural_networks(estimator, k_fold, test_imgs, train_imgs, save_model, model_name, normalize,
                            rebalance, seed)
    # Set defaults
    if model_name is None:
        model_name = 'model.pkl'
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    model_path = os.path.join(CONFIG.model_path, model_name)
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
        random.seed(seed)
        samples = list(CONFIG.all_samples)
        random.shuffle(samples)
        scores = []
        for k in range(k_fold):
            size = int(len(samples)/k_fold)
            test_imgs = samples[k*size:(k+1)*size]
            logger.debug('Running cross-validation %i/%i, testing on %s' % ((k+1), k_fold, str(sorted(test_imgs))))
            try:
                score = run_cross_validation(estimator, columns, test_imgs=test_imgs,
                                             save_model=(save_model and k == 0), model_name=model_name,
                                             normalize=normalize, rebalance=rebalance, seed=seed)
                scores.append(score)
            except EmptyDataSet:
                k_fold -= 1
        return scores

    cdb = ClusterDB()
    data_train, targets_train, data_test, targets_test = \
        get_train_test_data(cdb, test_imgs, train_imgs, columns, normalize, rebalance, seed)
    if estimator is None:
        estimator = load_estimator(model_path)
    else:
        logger.debug('Fitting estimator...')
        estimator.fit(data_train, targets_train)
        if save_model:
            logger.debug('Saving estimator in %s' % model_path)
            joblib.dump(estimator, model_path)
    logger.debug('Testing estimator...')
    score = estimator.score(data_test, targets_test)
    logger.info("Estimator %s has a score of %s" % (str(estimator), str(score)))
    return score


def predict_and_show(estimator, img_num, complete_df, width, height):
    assert width * height == len(complete_df)
    logger.info("Predicting neural network results on image %i ..." % img_num)
    img_out_path = os.path.join(CONFIG.model_path, "prediction_image%i.jpg" % img_num)
    array = np.zeros((width, height, 3), 'uint8')
    prediction = estimator.predict(complete_df)
    prediction = prediction.reshape((width, height))
    prediction -= 3  # predict 0 or 1, instead of 3 or 4
    array[..., 0] = 255*(1-prediction)
    array[..., 1] = 255*prediction
    img = Image.fromarray(array)
    img.save(img_out_path)


def run_neural_networks(nn, k_fold, test_imgs, train_imgs, save_model, model_name, normalize, rebalance, seed):
    return nn.run_cross_validation(k_fold, test_imgs, train_imgs, save_model, model_name, normalize, rebalance, seed)


if __name__ == '__main__':
    # # ######################## Example 1.1: test new sklearn model ########################
    # from sklearn.svm import SVC
    # from sklearn.neighbors import KNeighborsClassifier
    # _estimator = SVC()
    # _test_imgs = 16
    # _columns = ['M22', 'M23', 'M24', 'M32', 'M34', 'M42', 'M43']
    # _k_fold = 3
    #
    # score = run_cross_validation(_estimator, _columns, test_imgs=_test_imgs, train_imgs=[1, 4, 17], save_model=True,
    #                              model_name='svc')

    # ######################## Example 1.2: test saved sklearn model ########################
    # from sklearn.svm import SVC
    # from sklearn.neighbors import KNeighborsClassifier
    # _estimator = SVC()
    # _test_imgs = 16
    # _columns = ['M22', 'M23', 'M24', 'M32', 'M34', 'M42', 'M43']
    # _k_fold = 3
    #
    # _score = run_cross_validation(None, _columns, test_imgs=14, model_name='svc')
    # print(_score)

    # ######################## Example 1.3: show saved sklearn model results ########################
    # _estimator = load_estimator(os.path.join(CONFIG.model_path, 'svc.pkl'))
    # _img_num = 10
    # _complete_df = pd.read_csv(os.path.join(CONFIG.backup_data_path, 'samples', 'Sample10', 'concatenation.csv'))
    # _columns = ['M22', 'M23', 'M24', 'M32', 'M34', 'M42', 'M43']
    # _complete_df = _complete_df[_columns]
    # with open(os.path.join(CONFIG.backup_data_path, 'samples', 'Sample10', '.shape'), 'r') as in_f:
    #     _width, _height = [int(i) for i in in_f.readline()[1:-1].split(',')]
    #
    # predict_and_show(_estimator, _img_num, _complete_df, _width, _height)

    # # ######################## Example 2.1: test neural network ########################
    # _columns = ['M22', 'M23', 'M24', 'M32', 'M34', 'M42', 'M43', 'mean', 'std']
    # _layers = [9, 50, 2]  # 9=len(_columns), 2=len({3, 4})
    # _nn = NeuralNetwork(_columns, _layers, maxfun=5)
    # score = _nn.run_cross_validation(save_model=True)
    # print(score)

    # # ######################## Example 2.2: show results af a saved neural network ########################
    # _nn = NeuralNetwork(model_name='neural_networks')
    # _complete_df = pd.read_csv(os.path.join(CONFIG.backup_data_path, 'samples', 'Sample10', 'concatenation.csv'))
    # with open(os.path.join(CONFIG.backup_data_path, 'samples', 'Sample10', '.shape'), 'r') as in_f:
    #     _width, _height = [int(i) for i in in_f.readline()[1:-1].split(',')]
    # _nn.predict_and_show(10, _complete_df, _width, _height)
    pass
