import unittest
import os
import numpy as np
from numpy.linalg import svd
from matplotlib import pyplot as plt

from database import ClusterDB
from configuration import CONFIG
from visualization import Visualization


class VisualizationTest(unittest.TestCase):
    cdb = None
    v = None

    @classmethod
    def setUpClass(cls):
        cls.cdb = ClusterDB()
        cls.v = Visualization(cls.cdb)

    def test_mij_histograms(self, couples=None):
        if couples is None:
            couples = []
            for col_name in CONFIG.relevant_columns:
                if col_name.startswith("M") and len(col_name) == 3:
                    couples.append((int(col_name[1]), int(col_name[2])))

        histogram_dir = os.path.join(CONFIG.visualization_path, "Histogrammes")
        for file_name in os.listdir(histogram_dir):
            os.remove(os.path.join(histogram_dir, file_name))
        self.assertEqual(0, len(os.listdir(histogram_dir)))
        for (i, j) in couples:
            self.v.mij_histogram(i, j, out_path=os.path.join(histogram_dir, 'M%i%i.png' % (i, j)))
        self.assertEqual(len(couples), len(os.listdir(histogram_dir)))

    def test_outliers(self):
        c = self.cdb.centers[CONFIG.learning_columns]
        mean_s = c.mean()
        c = c.transpose().apply(lambda s: s - mean_s)
        # Uncomment to see outliers appear
        # plt.plot(np.sqrt([sum(np.square(c[i])) for i in c.columns]))
        # plt.show()
        outliers = [i for i in c.columns if np.sqrt(sum(np.square(c[i]))) > 0.4]
        self.assertEqual([22, 35], outliers)

    def test_svd(self, without_outliers=True):
        outliers = [22, 35]
        base_names = ["eigenvalues", "eigenvectors", "dots"]
        if without_outliers:
            centers = self.cdb.centers.loc[set(range(len(self.cdb.centers))).difference(outliers), :]
            out_names = [base+"_without_outliers.png" for base in base_names]
        else:
            centers = self.cdb.centers
            out_names = [base+"_with_outliers.png" for base in base_names]
        centers = centers[CONFIG.relevant_columns]

        out_paths = [os.path.join(CONFIG.visualization_path, "SVD", name) for name in out_names]
        for path in out_paths:
            if os.path.isfile(path):
                os.remove(path)
        nb_files = len(os.listdir(os.path.join(CONFIG.visualization_path, "SVD")))

        u, s, v = svd(centers[CONFIG.learning_columns].transpose())
        k = len(CONFIG.learning_columns)  # dimension
        plt.figure(1)
        index = np.arange(k+1)
        plt.plot(np.arange(1, k+1), np.sqrt(s/s[0]))
        plt.plot([0.9]*(k+1))
        plt.plot(index, [sum(s[:i])/sum(s) for i in index])
        plt.savefig(out_paths[0])
        self.assertEqual(nb_files + 1, len(os.listdir(os.path.join(CONFIG.visualization_path, "SVD"))))
        plt.clf()

        fig, ax = plt.subplots()
        p = 4
        ind = np.arange(k)
        width = 0.7/p
        for i in range(p):
            ax.bar(ind+i*width, u.transpose()[i], width, color=str(i/p))
        ax.set_xticks(ind + 0.35)
        ax.set_xticklabels(CONFIG.learning_columns)
        plt.savefig(out_paths[1])
        self.assertEqual(nb_files + 2, len(os.listdir(os.path.join(CONFIG.visualization_path, "SVD"))))

        plt.clf()
        centers4 = centers[centers['diagnosis'] == 4][CONFIG.learning_columns]
        centers3 = centers[centers['diagnosis'] == 3][CONFIG.learning_columns]
        svd_c_4 = np.dot(u.transpose()[:2], centers4.transpose())
        svd_c_3 = np.dot(u.transpose()[:2], centers3.transpose())
        plt.plot(svd_c_4[0], svd_c_4[1], 'go')
        plt.plot(svd_c_3[0], svd_c_3[1], 'ro')

        plt.savefig(out_paths[2])
        self.assertEqual(nb_files + 3, len(os.listdir(os.path.join(CONFIG.visualization_path, "SVD"))))
