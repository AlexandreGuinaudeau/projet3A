import unittest
from database import ClusterDB
from configuration import CONFIG
from svm import SVM


class SVMTest(unittest.TestCase):
    cdb = None
    svm = None

    @classmethod
    def setUpClass(cls):
        cls.cdb = ClusterDB()
        cls.svm = SVM(cls.cdb)

    def test_cross_validation1(self, normalize=True):
        cum_score = []
        cum_sum = 0
        for img in CONFIG.all_samples:
            weight = len(self.cdb.filter(img_nums=img))
            score = self.svm.cross_validate(img, columns=CONFIG.relevant_columns, normalize=normalize)
            if score is not None:
                cum_score.append(score*weight)
                cum_sum += weight
        print("Average:", sum(cum_score)/cum_sum, sum(cum_score), cum_sum)

    def test_cross_validation2(self, normalize=True):
        cum_score = []
        cum_sum = 0
        for img1 in CONFIG.all_samples:
            print('Testing with image %i...' % img1)
            for img2 in CONFIG.all_samples:
                weight = len(self.cdb.filter(img_nums=[img1, img2]))
                score = self.svm.cross_validate([img1, img2], columns=CONFIG.relevant_columns, normalize=normalize)
                if score is not None:
                    cum_score.append(score*weight)
                    cum_sum += weight
        print("Average:", sum(cum_score)/cum_sum, sum(cum_score), cum_sum)


