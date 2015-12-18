import unittest
import os
import logging
import time
import pandas as pd

from mueller_matrix import MuellerMatrix, MuellerImage
from utils import replace_if_condition, prepare_sample_dir

# Set sample nb
sample_nb = 7

data_dir = os.path.join(os.path.realpath(__name__), "..", "..", "data")
sample_dir = os.path.join(data_dir, "Sample%i" % sample_nb)
in_sample_path = os.path.join(sample_dir, "sample_first_last.csv")
# npos_path = os.path.join(os.path.realpath(__name__), "..", "data", "NPOS.csv")
raw_xy_path = os.path.join(data_dir, "XY.csv")
data_xy_path = os.path.join(sample_dir, "XY.csv")
raw_pixtup_path = os.path.join(data_dir, "PixTup.csv")
data_pixtup_path = os.path.join(sample_dir, "PixTup%i.csv" % sample_nb)
npos_index = {1: 0, 2: 13787, 3: 38674, 4: 43118, 5: 51690, 6: 55047, 7: 55859, 8: 70072, 9: 93299, 10: 101594,
              11: 107798, 12: 122664, 13: 129064, 14: 132474, 15: 137918, 16: 143697, 17: 146779, 18: 163041}
logging.basicConfig(level=logging.DEBUG)


class TestMuellerMatrix(unittest.TestCase):
    matrix = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):
        return
        cls.logger.setLevel(logging.DEBUG)
        if not os.path.isdir(os.path.join(sample_dir, "images")):
            prepare_sample_dir(sample_dir)
        # img_dict = {}
        # img_in_sample_path = in_sample_path if os.path.isfile(in_sample_path) else None
        # for i in range(1, 5):
        #     for j in range(1, 5):
        #         in_path = os.path.join(data_path, "csv", "M%i%i.csv" % (i, j))
        #         img_dict[i, j] = MuellerImage(pd.read_csv(in_path, header=None), in_sample_path=img_in_sample_path)
        #         cls.logger.debug("Loaded image M%i%i" % (i, j))
        img_dir = os.path.join(sample_dir, "csv")
        cls.matrix = MuellerMatrix(img_dir, logger_level=logging.DEBUG)
        if cls.matrix.save_in_sample_convex_hull(in_sample_path, overwrite=False):
            cls.logger.debug("Sample zone found and saved")

    def save_all_new_images(self):
        new = 0
        for i in range(1, 5):
            for j in range(1, 5):
                if self.test_save(i, j, overwrite=False):
                    new += 1
        print("Saved %i new images" % new)

    @unittest.skip
    def test_init(self):
        self.save_all_new_images()

    @unittest.skip
    def test_save(self, i=1, j=1, overwrite=True, rgb=False):
        out_path = os.path.join(sample_dir, "images", "M%i%i.png" % (i, j))
        if os.path.isfile(out_path):
            if not overwrite:
                return False
            os.remove(out_path)
        if rgb:
            self.matrix[i, j].save_rgb(out_path)
        else:
            self.matrix[i, j].save(out_path)
        # print("Saved image M%i%i" % (i, j))
        self.assertTrue(os.path.isfile(out_path))
        return True

    @unittest.skip
    def test_sample_detection(self):
        g = self.matrix._save_in_sample_convex_hull(in_sample_path, overwrite=True)
        self.assertTrue(g)

    @unittest.skip
    def test_find_sample_nb(self):
        df = pd.read_csv(raw_pixtup_path, header=None)
        df = df.iloc[:, 1:17]
        found = None
        for i in range(1, 18):
            if self._matches_sample(df, npos_index[i]):
                print("Found: %i" % i)
                assert found is None
                found = i
            else:
                print("Not %i" % i)
        self.assertEqual(found, sample_nb)

    def _matches_sample(self, df, first_index):
        df = df[first_index:first_index+1]
        for index, row in df.iterrows():
            for c, cell in enumerate(row):
                mat_df = self.matrix[int(c / 4) + 1, (c % 4) + 1].df
                if c > 0:
                    mat_df = mat_df.mul(copy)
                copy = mat_df.applymap(replace_if_condition(lambda x: x == cell, 1, 0))
                if max(copy.max()) == min(copy.min()):
                    print(c)
                    return False
        return True

    @unittest.skip
    def test_save_centered_pixtup(self):
        pixtup_df = pd.read_csv(raw_pixtup_path, header=None)
        pixtup_df = pixtup_df.iloc[npos_index[sample_nb]:npos_index[sample_nb+1], 0:17]
        xy_df = pd.read_csv(raw_xy_path, header=None)
        xy_df = xy_df.iloc[npos_index[sample_nb]:npos_index[sample_nb+1]]
        for k in range(1, 17):
            print("k =", k)
            centered_df = self.matrix[int((k-1)/4)+1, ((k-1) % 4) + 1].centered_df
            for index, row in pixtup_df.iterrows():
                index = index - npos_index[sample_nb]
                x, y = int(xy_df.iloc[index, 0]), int(xy_df.iloc[index, 1])
                pixtup_df.iat[index, k] = centered_df.iat[x, y]
        pixtup_df.to_csv(data_pixtup_path, header=None, index=False)

    @unittest.skip
    def test_get_xy_area(self):
        """ Only keep first and last x indexes for each y line (convex areas)"""
        xy_df = pd.read_csv(raw_xy_path, header=None)
        xy_df = xy_df.iloc[npos_index[sample_nb]:npos_index[sample_nb+1]]
        x, y = int(xy_df.iloc[0, 0]), int(xy_df.iloc[0, 1])
        with open(data_xy_path, "w") as out_f:
            out_f.write("{x},{y}\n".format(x=x, y=y))
            x -= 1
            for index, row in xy_df.iterrows():
                index = index - npos_index[sample_nb]
                x1, y1 = int(xy_df.iloc[index, 0]), int(xy_df.iloc[index, 1])
                if y1 == y and x1 == x + 1:
                    x = x1
                    continue
                out_f.write("{x},{y}\n".format(x=x, y=y))
                out_f.write("{x},{y}\n".format(x=x1, y=y1))
                x, y = x1, y1
            out_f.write("{x},{y}\n".format(x=x, y=y))
        self.assertGreater(len(xy_df), len(pd.read_csv(data_xy_path, header=None)))
