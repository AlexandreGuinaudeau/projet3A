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
        c = self.db[(1, 2)]
        self.assertTrue(isinstance(c.img_num, int))
        self.assertEqual(16, len(c.center))
        self.assertEqual(1, c.variances[0])
        with self.assertRaises(OSError):
            # Lazy load
            print(c.df)
        self.assertTrue(isinstance(self.db[(1, 1)].df, pd.DataFrame))

    def test_image(self):
        self.assertTrue(isinstance(self.db[1], Image))
        self.assertEqual(2, self.db[1].nb_clusters)
