import os
import random
import pandas as pd
from configuration import CONFIG


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
        for Mij in ['M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24', 'M31', 'M32', 'M33', 'M34',
                    'M41', 'M42', 'M43', 'M44']:
            self._df[Mij] = self._df['M11'] * self._df[Mij]
        self._center = self._df[['M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                                 'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']].mean()
        self._variances = self._df[['M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                                    'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']].var()

    @property
    def center(self):
        if self._center is None:
            self._center = self.df[['M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                                    'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']].mean()
        return self._center

    @property
    def variances(self):
        if self._variances is None:
            self._variances = self.df[['M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                                       'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']].var()
        return self._variances

    def _load_df(self):
        self._df = pd.read_csv(self.file_path)

    def apply_cluster(self, array, nb_clusters):
        #array : output, red=Ill, green=Safe
        df=self._df[["x"]["y"]]
        if self.diagnosis == 3:
            for index, row in df.iterrows():
                x, y = row
                array[x, y] = (0,int(255*(self.cluster_num+5)/(nb_clusters+5)),0)
        elif self.diagnosis == 4:
            for index, row in df.iterrows():
                x, y = row
                array[x, y] = (int(255*(self.cluster_num+5)/(nb_clusters+5)),0,0)

    def save(self, out_path):
        height = 422
        width = 560
        array = np.zeros((height, width, 3), 'uint8')
        array = self.apply_cluster(array, 1)
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

    def __init__(self, db_path=None, metadata_path=None, center_path=None, variances_path=None, normed=None):
        # Set default paths
        self._db_path = db_path if db_path is not None else CONFIG.db_path
        if center_path is None:
            center_path = CONFIG.center_path
        if variances_path is None:
            variances_path = CONFIG.variances_path
        if metadata_path is None:
            metadata_path = CONFIG.metadata_path
        self._normed = normed

        # Initialize
        self._clusters = {}
        self.centers = pd.read_csv(center_path) if os.path.isfile(center_path) else None
        self.variances = pd.read_csv(variances_path) if os.path.isfile(variances_path) else None
        self._load_metadata(metadata_path)

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

    def _load_metadata(self, metadata_path):
        df = pd.read_csv(metadata_path, header=None,
                         names=['img_num', 'diagnosis', 'cluster_num', 'file_name'])
        for index, row in df.iterrows():
            img_num, diagnosis, cluster_num, file_name = row
            file_path = os.path.join(self._db_path, file_name)
            center_s = None
            variances_s = None
            if self.centers is not None:
                center_df = self.centers[self.centers['img_num'] == img_num][self.centers['cluster_num'] == cluster_num]
                center_df = center_df[center_df['diagnosis'] == diagnosis].transpose()
                center_s = center_df.iloc[:, 0]
            if self.variances is not None:
                vars_df = self.variances[self.variances['img_num'] == img_num]
                vars_df = vars_df[vars_df['cluster_num'] == cluster_num][vars_df['diagnosis'] == diagnosis].transpose()
                variances_s = vars_df.iloc[:, 0]
            self._clusters[(img_num, diagnosis, cluster_num)] = \
                Cluster(img_num, diagnosis, cluster_num, file_path, center=center_s, variances=variances_s,
                        normed=self._normed)

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

    def filter(self, rebalance=False, seed=None, *, img_nums=None, diagnosis=None, cluster_nums=None, columns=None):
        """
        Selects matching clusters in the database.

        Parameters
        ==========

        rebalance: bool, guarantees to have the same number of each diagnosis after filtering
        seed: int, sets the random number to rebalance the results
        img_nums: int or array, images to keep
        diagnosis: int or array, diagnosis to keep
        cluster_nums: int or array, clusters to keep
        columns: str or list of str, columns to keep

        Returns
        =======
        centers: The Dataframe of the centers matching the filter criteria
        """
        centers = self.centers.copy()
        if isinstance(img_nums, int):
            img_nums = [img_nums]
        if isinstance(diagnosis, int):
            diagnosis = [diagnosis]
        if isinstance(cluster_nums, int):
            cluster_nums = [cluster_nums]
        if isinstance(columns, str):
            columns = [columns]
        if img_nums is not None:
            centers = centers[centers.img_num.isin(img_nums)]
        if diagnosis is not None:
            centers = centers[centers.diagnosis.isin(diagnosis)]
        if cluster_nums is not None:
            centers = centers[centers.cluster_nums.isin(cluster_nums)]
        if columns is not None:
            centers = centers[columns]
        if rebalance:
            random.seed(seed)
            centers_3 = centers[centers['diagnosis'] == 3]
            centers_4 = centers[centers['diagnosis'] == 4]
            if not (len(centers_3) and len(centers_4)):
                centers.drop(centers.index)
            if len(centers_3) < len(centers_4):
                idx = list(centers_4.index)
                random.shuffle(idx)
                centers.drop(idx[len(centers_3):], inplace=True)
            if len(centers_3) > len(centers_4):
                idx = list(centers_3.index)
                random.shuffle(idx)
                centers.drop(idx[len(centers_4):], inplace=True)
        if len(centers):
            return centers
        else:
            return None  # No center matches the filter.


    def save(self, out_path, *, img_nums=None, diagnosis=None, cluster_nums=None):
        arrays = {}
        height = 422
        width = 560
        temp_df = filter(img_nums, diagnosis, cluster_nums)
        for (i, d, c) in set(temp_df.keys()):
            if i not in arrays.keys():
                arrays[i] = np.zeros((height, width, 3), 'uint8')
            arrays[i] = temp_df[i,d,c].apply_cluster(temp_df[i,d,c], arrays[i], max(self.centers['cluster_num']))
        for i, array in arrays.items():
            img = Image.fromarray(array)
            img.save(out_path+str(i))