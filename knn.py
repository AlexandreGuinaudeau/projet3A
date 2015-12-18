from sklearn.neighbors import KNeighborsClassifier
from utils import get_train_test_data


class Knn:
    def __init__(self, cdb, n_neighbors=5, weights='distance'):
        self.cdb = cdb
        self._knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def cross_validate(self, img_nums=None, columns=None, normalize=False, rebalance=True, seed=None):
        data_train, targets_train, data_test, targets_test = \
            get_train_test_data(self.cdb, img_nums, columns, normalize, rebalance, seed)

        self._knn.fit(data_train, targets_train)
        score = self._knn.score(data_test, targets_test)
        return score
