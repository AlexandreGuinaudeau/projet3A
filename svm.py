import numpy as np
from sklearn import svm
from configuration import CONFIG


class SVM:
    def __init__(self, cdb, svm_type='Kernel'):
        self.cdb = cdb
        if svm_type == 'Kernel':
            self._svm = svm.SVC(gamma=.2)
        else:
            self._svm = None
        self._clf = None

    def cross_validate(self, img_nums=None, columns=None, normalize=False, rebalance=True, seed=None):
        if isinstance(img_nums, int):
            img_nums = [img_nums]

        train_centers = self.cdb.filter(rebalance=rebalance, seed=seed,
                                        img_nums=CONFIG.all_samples.difference(img_nums), columns=columns)
        if train_centers is None:
            return None
        data_train = train_centers[CONFIG.relevant_columns]
        if normalize:
            # Center and norm the data (mean=0, variance=1)
            data_train = data_train.apply(lambda s: s - data_train.mean(), axis=1)
            data_train = data_train.apply(lambda s: s/np.sqrt(data_train.var()), axis=1)
        targets_train = train_centers['diagnosis']

        test_centers = self.cdb.filter(img_nums=img_nums, columns=columns)
        if test_centers is None:
            return None
        data_test = test_centers[CONFIG.relevant_columns]
        if normalize:
            # Center and norm, based on train data
            data_test = data_test.apply(lambda s: s - data_train.mean(), axis=1)
            data_test = data_test.apply(lambda s: s/np.sqrt(data_train.var()), axis=1)
        targets_test = test_centers['diagnosis']

        self._svm.fit(data_train, targets_train)
        score = self._svm.score(data_test, targets_test)
        return score
