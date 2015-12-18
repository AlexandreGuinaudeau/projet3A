from sklearn import svm
from utils import get_train_test_data


class SVM:
    def __init__(self, cdb, svm_type='Kernel'):
        self.cdb = cdb
        if svm_type == 'Kernel':
            self._svm = svm.SVC(gamma=.2)
        else:
            self._svm = None
        self._clf = None

    def cross_validate(self, img_nums=None, columns=None, normalize=False, rebalance=True, seed=None):
        data_train, targets_train, data_test, targets_test = \
            get_train_test_data(self.cdb, img_nums, columns, normalize, rebalance, seed)

        self._svm.fit(data_train, targets_train)
        score = self._svm.score(data_test, targets_test)
        return score
