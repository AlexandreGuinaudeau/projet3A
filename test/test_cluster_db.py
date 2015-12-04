import unittest
import os
import pandas as pd
from database.cluster_db import Cluster, Image, ClusterDB

db_path = os.path.realpath(os.path.join(__name__, "..", "test_db"))
metadata_path = os.path.join(db_path, "test_metadata.csv")


class ClusterDBTest(unittest.TestCase):
    db = None

    @classmethod
    def setUpClass(cls):
        cls.db = ClusterDB(db_path, metadata_path)

    def test_init(self):
        self.assertTrue(True)

    def test_cluster(self):
        self.assertTrue(isinstance(self.db[(1, 2)], Cluster))
        c12 = self.db[(1, 2)]
        self.assertTrue(isinstance(c12.img_num, int))
        with self.assertRaises(OSError):
            # Lazy load
            print(c12.df)
        c11 = self.db[(1, 1)]
        print(c11)
        print(c11.center)
        print(c11.df)
        self.assertEqual(16, len(c11.center))
        self.assertEqual(0, c11.variances['M12'])
        c11.norm()
        self.assertNotEqual(0, c11.variances['M12'])
        self.assertTrue(isinstance(c11.df, pd.DataFrame))

    def test_image(self):
        self.assertTrue(isinstance(self.db[1], Image))
        self.assertEqual(2, self.db[1].nb_clusters)