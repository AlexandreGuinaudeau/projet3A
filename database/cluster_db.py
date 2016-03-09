import os
import random
import pandas as pd
import numpy as np
from PIL import Image
from configuration import CONFIG
from utils import load_df


class Cluster:
    """
    Represents a group of 'similar' points.
    Call norm() to multiply all Mij by M11.

    Attributes
    ==========

    cluster_num: int >=1, an id, unique among the clusters of this image with this diagnosis.
        There is no hierarchy among clusters of a (image, diagnosis) tuple.
    img_num: int >=1, the number of the image the points come from
    diagnosis: 3 or 4, the diagnosis of all the points of the cluster.
    file_path: str, the path to the csv file storing the data of the points from this cluster.
    df: pd.Dataframe, the Mij values of the points. It is loaded when called for the first time.
    center: pd.Series, the Mij values of the center of the cluster (mean value of each Mij)
    variances: pd.Series, for each Mij, the variance of the points of the cluster along this axis.
    """

    def __init__(self, img_num, diagnosis, cluster_num, file_path, center=None, variances=None, normed=None):
        self.img_num = img_num
        self.diagnosis = diagnosis
        self.cluster_num = cluster_num
        self.file_path = file_path
        self._df = None
        self._center = center
        self._variances = variances
        self._normed = normed if normed is not None else os.path.isfile(CONFIG.is_normed_path)

    def __str__(self):
        return "<Cluster (%i, %i, %i)>" % (self.img_num, self.diagnosis, self.cluster_num)

    @property
    def df(self):
        if self._df is None:
            self._load_df()
        return self._df

    def norm(self):
        if self._normed:
            return
        self._normed = True
        for Mij in CONFIG.all_mij_columns[1:]:
            self._df[Mij] = self._df['M11'] * self._df[Mij]
        self._center = self._df[CONFIG.all_mij_columns].mean()
        self._variances = self._df[CONFIG.all_mij_columns].var()

    @property
    def center(self):
        if self._center is None:
            self._center = self.df[CONFIG.all_mij_columns].mean()
        return self._center

    @property
    def variances(self):
        if self._variances is None:
            self._variances = self.df[CONFIG.all_mij_columns].var()
        return self._variances

    def _load_df(self):
        self._df = pd.read_csv(self.file_path)

    def apply_cluster(self, array, nb_clusters):
        # array : output, red=Ill, green=Safe
        df = self.df[["x", "y"]]
        if self.diagnosis == 4:
            for index, row in df.iterrows():
                x, y = row
                array[x-1, y-1] = (0, int(255*(self.cluster_num+5)/(nb_clusters+5)), 0)
        elif self.diagnosis == 3:
            for index, row in df.iterrows():
                x, y = row
                array[x-1, y-1] = (int(255*(self.cluster_num+5)/(nb_clusters+5)), 0, 0)

    def save(self, out_path):
        height = CONFIG.height
        width = CONFIG.width
        array = np.zeros((height, width, 3), 'uint8')
        self.apply_cluster(array, 1)
        img = Image.fromarray(array)
        img.save(out_path)


class ClusterDB:
    """
    Represents all the data.
    To access only centers of clusters or variances, use cdb.centers and cdb.variances.

    Examples
    ========

    cdb = ClusterDB(CONFIG.db_path, CONFIG.metadata_path)
    cdb.filter(img_num=1)        # Keep only elements from image 1
    for cluster in cdb:          # Iterate on clusters form image 1
        ...
    cdb.unfilter()               # Undo all previous filters.
    cluster1_4_2 = cdb[1, 4, 2]  # Access a given cluster.
    """

    def __init__(self):
        # Set default paths
        self._db_path = CONFIG.data_path
        center_path = CONFIG.center_path
        self._data_path = CONFIG.current_data_path

        # Initialize
        self._clusters = {}
        self.centers = pd.read_csv(center_path) if os.path.isfile(center_path) else None

    def __str__(self):
        info = "<ClusterDB: %i Clusters>\n" % self.nb_clusters
        for key in self.sorted_cluster_keys:
            info += "\t" + str(key) + "\n"
        info += "</ClusterDB>"
        return info

    def __getitem__(self, item):
        if isinstance(item, tuple) and item in self._clusters.keys():
            return self._clusters[item]
        return self._clusters[self.sorted_cluster_keys.__getitem__(item)]

    @property
    def nb_clusters(self):
        return len(self._clusters)

    @property
    def clusters(self):
        return self._clusters.values()

    @property
    def sorted_cluster_keys(self):
        return sorted(self._clusters.keys())

    def filter_clusters(self, *, img_nums=None, diagnosis=None, cluster_nums=None):
        """
        Selects matching clusters in the database.

        Parameters
        ==========

        img_nums: int or array, images to keep in database
        diagnosis: int or array, diagnosis to keep in database
        cluster_nums: int or array, clusters to keep in database

        Returns
        =======
        clusters: The dictionary of the filtered clusters {(img_num, diagnosis, cluster_num) => Cluster, ...}
        """
        filtered_clusters = self._clusters.copy()
        if isinstance(img_nums, int):
            img_nums = [img_nums]
        if isinstance(diagnosis, int):
            diagnosis = [diagnosis]
        if isinstance(cluster_nums, int):
            cluster_nums = [cluster_nums]
        for (i, d, c) in set(filtered_clusters.keys()):
            if img_nums is not None and i not in img_nums:
                filtered_clusters.pop((i, d, c))
                continue
            if diagnosis is not None and d not in diagnosis:
                filtered_clusters.pop((i, d, c))
                continue
            if cluster_nums is not None and c not in cluster_nums:
                filtered_clusters.pop((i, d, c))
        return filtered_clusters

    def filter(self, rebalance=False, seed=None, *, img_nums=None, diagnosis=None, columns=None):
        """
        Selects matching points in the database.

        Parameters
        ==========

        img_nums: int or array, images to keep
        diagnosis: int or array, diagnosis to keep
        columns: str or array, columns to keep
        rebalance: bool, whether the result should have as many points of each diagnosis
        seed: int or None, seed of the random generator

        Returns
        =======
        df: The pd.Dataframe of the points matching the filter
        """
        df = load_df()
        if isinstance(img_nums, int):
            img_nums = [img_nums]
        if isinstance(diagnosis, int):
            diagnosis = [diagnosis]
        if isinstance(columns, str):
            columns = [columns]

        if img_nums is not None:
            df = df[df['img_num'].isin(img_nums)]
        if diagnosis is not None:
            df = df[df['diagnosis'].isin(diagnosis)]
        if rebalance:
            random.seed(seed)
            points3 = df[df['diagnosis'] == 3]
            points4 = df[df['diagnosis'] == 4]
            if len(points3) == 0 or len(points4) == 0:
                # Cannot rebalance
                return None
            if len(points3) < len(points4):
                idx = list(points4.index)
                random.shuffle(idx)
                df.drop(idx[len(points3):], inplace=True)
            else:
                idx = list(points3.index)
                random.shuffle(idx)
                df.drop(idx[len(points4):], inplace=True)
        if columns is not None:
            df = df[df.columns.intersection(columns)]
        return df

    def save(self, *, img_nums=None, diagnosis=None, cluster_nums=None):
        arrays = {}
        temp_clusters = self.filter_clusters(img_nums=img_nums, diagnosis=diagnosis, cluster_nums=cluster_nums)
        for (i, d, c) in self.sorted_cluster_keys:
            if i not in arrays.keys():
                arrays[i] = np.zeros((CONFIG.height, CONFIG.width, 3), 'uint8')
            temp_clusters[i, d, c].apply_cluster(arrays[i], max(self.centers['cluster_num']))
        for i, array in arrays.items():
            img = Image.fromarray(array)
            out_path = os.path.join(CONFIG.visualization_path, "clusters", "image%i.png" % i)
            img.save(out_path)
