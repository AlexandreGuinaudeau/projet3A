import unittest
import random
from database import ClusterDB
from configuration import CONFIG
from knn import Knn


class SVMTest(unittest.TestCase):
    cdb = None
    knn = None

    @classmethod
    def setUpClass(cls):
        cls.cdb = ClusterDB()
        cls.knn = Knn(cls.cdb, n_neighbors=4, weights='uniform')

    def test_cross_validation(self, test_size=1, normalize=True, rebalance=True, seed=None):
        cum_score = []
        cum_sum = 0
        samples = list(CONFIG.all_samples)
        random.seed(seed)
        random.shuffle(samples)
        for i in range(int(len(samples)/test_size)):
            img_nums = samples[i*test_size:(i+1)*test_size]
            weight = len(self.cdb.filter(img_nums=img_nums))
            score = self.knn.cross_validate(img_nums,
                                            columns=CONFIG.relevant_columns,
                                            normalize=normalize,
                                            rebalance=rebalance,
                                            seed=seed)
            if score is not None:
                cum_score.append(score*weight)
                cum_sum += weight
        print("Average:", sum(cum_score)/cum_sum, sum(cum_score), cum_sum)
