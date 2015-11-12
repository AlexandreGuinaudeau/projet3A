import time
import logging
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull, Delaunay

from utils import replace_if_condition
logging.basicConfig(level=logging.DEBUG)


class MuellerException(Exception):
    pass


class MuellerImage:
    def __init__(self, df, in_sample_path=None):
        self.df = df
        self.width = len(df.columns)
        self.height = len(df)
        if in_sample_path is None:
            self.min = df.values.min()
            self.max = df.values.max()
            self.mean = None
            self.variance = None
            self.area = self.width*self.height
            self.centered_df = None
            return
        in_sample_df = pd.read_csv(in_sample_path, header=None)
        self.min = 10
        self.max = -10
        self.mean = 0
        self.variance = 0
        self.area = 0
        for x, row in df.iterrows():
            try:
                min_y, max_y = int(in_sample_df.iloc[x, 0]), int(in_sample_df.iloc[x, 1])
            except ValueError:
                continue
            row = row[min_y:max_y]
            self.min = min(self.min, min(row))
            self.max = max(self.max, max(row))
            self.mean += sum(row)
            self.area += len(row)
        self.mean /= self.area
        centered_df = df.applymap(lambda t: t-self.mean)
        var_df = centered_df.applymap(lambda t: t*t)
        for x, row in var_df.iterrows():
            try:
                min_y, max_y = int(in_sample_df.iloc[x, 0]), int(in_sample_df.iloc[x, 1])
            except ValueError:
                continue
            row = row[min_y:max_y]
            self.variance += sum(row)
        self.variance /= self.area
        import math
        centered_df = centered_df.applymap(lambda t: t/math.sqrt(self.variance))
        self.centered_df = centered_df

    def __str__(self):
        if self.mean is None:
            return "<MuellerImage %ix%ipx" % (self.width, self.height)
        return "<MuellerImage %ix%ipx [%f, %f] mean:%f variance:%f>" \
               % (self.width, self.height, self.min, self.max, self.mean, self.variance)

    def save(self, out_path):
        img = Image.fromarray(np.array(self.df
                                       .applymap(lambda x: int((x - self.min)*255 /
                                                               (self.max - self.min)))
                                       .as_matrix(),
                                       dtype=np.uint8))
        img.save(out_path)


class MuellerMatrix:
    logger = logging.getLogger(__name__)

    def __init__(self, img_dict, logger_level=logging.INFO):
        self.logger.setLevel(logger_level)
        self.img_dict = img_dict
        for i in range(1, 5):
            for j in range(1, 5):
                assert ((i, j) in img_dict.keys()), "There is no image with key (%i, %i) in %s" % (i, j, str(img_dict))
                assert isinstance(self[i, j], MuellerImage), "The element at (%i, %i) is no MatrixImage: %s" % \
                                                             (i, j, str(self[i, j]))
        self.width = self[1, 1].width
        self.height = self[1, 1].height

    def __getitem__(self, item):
        if isinstance(item, tuple) and item in self.img_dict.keys():
            return self.img_dict.__getitem__(item)
        if isinstance(item, int) and item in range(1, 5):
            return self.img_dict.__getitem__((item, item))
        raise MuellerException("%s is not a valid key for MuellerMatrix" % str(item))

    def __str__(self):
        return "<MuellerMatrix %ix%ipx>" \
               % (self.width, self.height)

    def _log_time_msg(self, start, msg, max_time_laps=5):
        total = time.time() - start
        msg = "TIME LAPS %f s: %s" % (total, msg)
        if total > max_time_laps:
            self.logger.warning(msg)
        else:
            self.logger.debug(msg)

    def save_in_sample_convex_hull(self, out_path, overwrite=False):
        """
        To retrieve pixels that are in the sample, we proceed as following:
        1. Mark every pixel under a certain threshold as part of the sample (above for elt 1, under for elt 2, 3, 4)
        2. Get the convex envelop of the sample pixels
        3. Save it in a file
        :return: if overwritten
        """
        if not overwrite and os.path.isfile(out_path):
            return False
        start = time.time()
        # Reverse elt 1 where the sample is brighter than the background
        # Change each element so that its smallest value is 0
        df1 = self[1].df.applymap(lambda x: self[1].max - x)
        df2 = self[2].df.applymap(lambda x: x - self[2].min)
        df3 = self[3].df.applymap(lambda x: x - self[3].min)
        df4 = self[4].df.applymap(lambda x: x - self[4].min)
        # Find the sample zone according to each diagonal element
        ceiled1 = self._in_sample_df(df1, self[1].max*0.85)
        ceiled2 = self._in_sample_df(df2, self[2].max*0.4)
        ceiled3 = self._in_sample_df(df3, self[3].max*0.4)
        ceiled4 = self._in_sample_df(df4, self[4].max*0.4)
        # Keep the sample zone common to all diagonal elements
        in_sample = ceiled1.add(ceiled2).add(ceiled3).add(ceiled4).applymap(replace_if_condition(lambda x: x < 4, 0, 1))
        self._log_time_msg(start, "Found isolated sample elements")
        start = time.time()
        # Get the convex envelop of the sample elements
        hull = self._find_convex_hull(in_sample)
        self._log_time_msg(start, "Found convex hull")
        start = time.time()
        self._save_convex_hull(in_sample, hull, out_path)
        self._log_time_msg(start, "Saved convex hull")
        return True

    @staticmethod
    def _save_convex_hull(in_sample, hull, out_path):
        with open(out_path, "w") as out_f:
            for x, row in in_sample.iterrows():
                first = None
                last = None
                for y, cell in enumerate(row):
                    if hull.find_simplex([x, y]) >= 0:
                        if first is None:
                            # 0 0 ... 0 1
                            first = y
                        continue
                    if first is None:
                        # 0 0 ... 0
                        continue
                    if last is None:
                        # 0 0 ... 0 1 ... 1 0
                        last = y
                        break
                out_f.write(str(first) + ", " + str(last) + "\n")

    @staticmethod
    def _in_sample_df(df, threshold):
        ceiled_df = df.applymap(replace_if_condition(lambda x: x < threshold, 0))
        ceiled_df = ceiled_df.applymap(replace_if_condition(lambda x: x > 0, 1))
        ceiled_df = ceiled_df.applymap(lambda x: 1 - x)
        return ceiled_df

    @staticmethod
    def _find_convex_hull(df):
        points = []
        for x, row in df.iterrows():
            for y, cell in enumerate(row):
                if cell:
                    points.append([x, y])
        hull = ConvexHull(np.array(points))
        hull = Delaunay(np.array([hull.points[i] for i in hull.vertices]))
        return hull

