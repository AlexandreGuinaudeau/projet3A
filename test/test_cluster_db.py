import unittest
import os
import pandas as pd
from database.cluster_db import Cluster, ClusterDB

db_path = os.path.realpath(os.path.join(__name__, "..", "test_db"))
metadata_path = os.path.join(db_path, "test_metadata.csv")
center_path = os.path.join(db_path, "test_centers.csv")


class ClusterDBTest(unittest.TestCase):
    db = None
    db_centers = None

    @classmethod
    def setUpClass(cls):
        cls.db = ClusterDB(db_path, metadata_path, "", "", False)
        cls.db_centers = ClusterDB(db_path, metadata_path, center_path, "", False)

    def test_init(self):
        self.assertTrue(True)

    def test_cluster(self):
        self.assertTrue(isinstance(self.db[(1, 3, 2)], Cluster))
        c12 = self.db[(1, 3, 2)]
        self.assertTrue(isinstance(c12.img_num, int))
        with self.assertRaises(OSError):
            # Lazy load
            print(c12.df)
        c11 = self.db[(1, 4, 1)]
        self.assertEqual(16, len(c11.center))
        self.assertEqual(0, c11.variances['M12'])
        c11.norm()
        self.assertNotEqual(0, c11.variances['M12'])
        self.assertTrue(isinstance(c11.df, pd.DataFrame))

    def test_centers(self):
        c11 = self.db_centers[(1, 4, 1)]
        self.assertEqual(16, len(c11.center))
        self.assertEqual(0, c11.variances['M12'])
        c11.norm()
        self.assertNotEqual(0, c11.variances['M12'])
        self.assertTrue(isinstance(c11.df, pd.DataFrame))

    def test_filter_database(self):
        self.assertEqual(7, self.db.nb_clusters)
        for cluster in self.db:
            self.assertTrue(isinstance(cluster, Cluster), cluster)
        filtered = self.db.filter(diagnosis=4)
        self.assertEqual(2, len(filtered))
        self.assertEqual(7, self.db.nb_clusters)
