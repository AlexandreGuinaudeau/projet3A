import time
import logging
import os
import numpy as np
import pandas as pd
from PIL import Image

from scipy.spatial import ConvexHull, Delaunay  # Works fine

from utils import replace_if_condition
from configuration import CONFIG
logging.basicConfig(level=logging.DEBUG)


class MuellerException(Exception):
    pass


class MuellerImage:
    def __init__(self, df, in_sample_path=None):
        self.df = df
        self.width = len(df.columns)
        self.height = len(df)
        # if in_sample_path is None:
        self.min = df.values.min()
        self.max = df.values.max()
        self.area = self.width*self.height
        # return
        # in_sample_df = pd.read_csv(in_sample_path, header=None)
        # self.min = 10
        # self.max = -10
        # self.area = 0
        # for x, row in df.iterrows():
        #     try:
        #         min_y, max_y = int(in_sample_df.iloc[x, 0]), int(in_sample_df.iloc[x, 1])
        #     except ValueError:
        #         continue
        #     row = row[min_y:max_y]
        #     self.min = min(self.min, min(row))
        #     self.max = max(self.max, max(row))

    def __str__(self):
        return "<MuellerImage %ix%ipx" % (self.width, self.height)

    def save(self, out_path):
        img = Image.fromarray(np.array(self.df.applymap(lambda x: int((x - self.min)*255 / (self.max - self.min)))
                                       .as_matrix(),
                                       dtype=np.uint8))
        img.save(out_path)

    def save_rgb(self, out_path):
        # img = self._get_01_image().convert("RGB")
        img = Image.fromarray(np.uint8(CONFIG.colormap(np.array(self.df.applymap(
            lambda x: ((x - self.min) / (self.max - self.min))).as_matrix()))*255))
        img.save(out_path)


class MuellerMatrix:
    logger = logging.getLogger(__name__)

    def __init__(self, sample_nb, file_name_format="M{i}{j}.csv", overwrite_metadata=False, logger_level=logging.INFO):
        self.logger.setLevel(logger_level)
        self.sample_nb = sample_nb

        self.sample_dir = os.path.join(CONFIG.data_path, "Sample%i" % sample_nb)
        self._csv_dir = os.path.join(self.sample_dir, "csv")
        self._metadata_dir = os.path.join(self.sample_dir, "metadata")
        self._file_name_format = file_name_format

        self._loaded_images = {}
        self._load_metadata(overwrite_metadata)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            i, j = item
        elif isinstance(item, int):
            i, j = (item, item)
        else:
            raise MuellerException("%s is not a valid key for MuellerMatrix" % str(item))
        if (i, j) not in self._loaded_images.keys():
            self._load_img(i, j)
        return self._loaded_images[(i, j)]

    def _load_metadata(self, overwrite):
        """
        If the folder doesn't exist, creates a 'metadata' folder containing:
        - XY3.csv, the list of (x, y) tuples delimiting a 3 (ill) area. Empty if no such area.
        - XY4.csv, the list of (x, y) tuples delimiting a 4 (healthy) area.
        - sample_first_last.csv, delimiting the sample area
        """
        if os.path.isdir(self._metadata_dir) and not overwrite:
            return
        self.logger.debug("Creating metadata folder for sample %i." % self.sample_nb)
        os.makedirs(self._metadata_dir)
        pix_tup_df = pd.read_csv(CONFIG.pixtup_path, header=None)
        pix_tup_df = pix_tup_df.iloc[CONFIG.npos_index[self.sample_nb]:CONFIG.npos_index[self.sample_nb+1]][0]
        xy_df = pd.read_csv(CONFIG.xy_path, header=None, names=["x", "y"])
        xy_df = xy_df.iloc[CONFIG.npos_index[self.sample_nb]:CONFIG.npos_index[self.sample_nb+1]]
        complete_df = pd.DataFrame(pd.concat([pix_tup_df, xy_df], axis=1))
        for type in [3, 4]:
            xy_type_path = os.path.join(self._metadata_dir, "XY%i.csv" % type)
            type_df = complete_df[complete_df[0] == type].reset_index()
            if not len(type_df):
                open(xy_type_path, "w").close()
                continue
            x, y = int(type_df.at[0, "x"]), int(type_df.at[0, "y"])
            with open(xy_type_path, "w") as out_f:
                out_f.write("{x},{y}\n".format(x=x, y=y))
                x -= 1
                for index, row in type_df.iterrows():
                    index = index
                    x1, y1 = int(type_df.at[index, "x"]), int(type_df.at[index, "y"])
                    if y1 == y and x1 == x + 1:
                        x = x1
                        continue
                    out_f.write("{x},{y}\n".format(x=x, y=y))
                    out_f.write("{x},{y}\n".format(x=x1, y=y1))
                    x, y = x1, y1
                out_f.write("{x},{y}\n".format(x=x, y=y))
            self.logger.info("XY%i file created at %s" % (type, xy_type_path))
        self._save_in_sample_convex_hull(os.path.join(self._metadata_dir, "sample_area.csv"))
        self.logger.info("Sample area file created.")

    def _load_img(self, i, j):
        in_path = os.path.realpath(os.path.join(self._csv_dir, self._file_name_format.format(i=i, j=j)))
        self._loaded_images[(i, j)] = MuellerImage(pd.read_csv(in_path, header=None),
                                                   os.path.join(self._metadata_dir, "sample_area.csv"))
        self.logger.debug("Loaded Image (%i,%i) from %s" % (i, j, in_path))

    def _save_in_sample_convex_hull(self, out_path, overwrite=False):
        """
        To retrieve pixels that are in the sample, we proceed as following:
        1. Mark every pixel under a certain threshold as part of the sample (above for elt 1, under for elt 2, 3, 4)
        2. Get the convex envelop of the sample pixels
        3. Save it in a file
        :return: if overwritten
        """
        if not overwrite and os.path.isfile(out_path):
            return False
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
        # Get the convex envelop of the sample elements
        hull = self._find_convex_hull(in_sample)
        self._save_convex_hull(in_sample, hull, out_path)
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
