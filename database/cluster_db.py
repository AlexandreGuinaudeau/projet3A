import os
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
    def __init__(self, cluster_num, img_num, diagnosis, file_path, center=None, variances=None):
        self.cluster_num = cluster_num
        self.img_num = img_num
        self.diagnosis = diagnosis
        self.file_path = file_path
        self._df = None
        self._center = center
        self._variances = variances
        self._normed = os.path.isfile(CONFIG.is_normed_path)

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
            self._df[Mij] = self._df['M11']*self._df[Mij]
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

    def save(self):
        # TODO (Pierre): code
        """ Saves the cluster as png. """


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
    def __init__(self, db_path=None, metadata_path=None, center_path=None, variances_path=None):
        # Set default paths
        self._db_path = db_path if db_path is not None else CONFIG.db_path
        if center_path is None:
            center_path = CONFIG.center_path
        if variances_path is None:
            variances_path = CONFIG.variances_path
        if metadata_path is None:
            metadata_path = CONFIG.metadata_path

        # Initialize
        self._clusters = {}
        self._clusters_backup = {}
        self.centers = pd.read_csv(center_path) if os.path.isfile(center_path) else None
        self.variances = pd.read_csv(variances_path) if os.path.isfile(variances_path) else None
        self._load_metadata(metadata_path)

    def __str__(self):
        info = "<ClusterDB: %i Clusters>\n" % self.nb_clusters
        for key in self.sorted_clusters:
            info += "\t" + str(key) + "\n"
        info += "</ClusterDB>"
        return info

    def __getitem__(self, item):
        if isinstance(item, tuple) and item in self._clusters.keys():
            return self._clusters[item]
        return self._clusters[self.sorted_clusters.__getitem__(item)]

    @property
    def nb_clusters(self):
        return len(self._clusters)

    @property
    def clusters(self):
        return self._clusters.values()

    @property
    def sorted_clusters(self):
        return sorted(self._clusters.keys())

    def _load_metadata(self, metadata_path):
        df = pd.read_csv(metadata_path, header=None,
                         names=['img_num', 'cluster_num', 'diagnosis', 'file_name'])
        for index, row in df.iterrows():
            img_num, diagnosis, cluster_num, file_name = row
            file_path = os.path.join(self._db_path, file_name)
            center = None
            if self.centers is not None:
                center = self.centers[self.centers['img_num'] == img_num][self.centers['cluster_num'] == cluster_num]
                center = pd.Series(center[center[diagnosis] == diagnosis])
            self._clusters[(img_num, diagnosis, cluster_num)] = \
                Cluster(img_num, cluster_num, diagnosis, file_path, center=center)
        self._clusters_backup = self._clusters.copy()

    def filter(self, *, img_nums=None, diagnosis=None, cluster_nums=None):
        """
        Selects matching clusters in the database.

        Parameters
        ==========

        img_nums: int or array, images to keep in database
        diagnosis: int or array, diagnosis to keep in database
        cluster_nums: int or array, clusters to keep in database
        """
        if isinstance(img_nums, int):
            img_nums = [img_nums]
        if isinstance(diagnosis, int):
            diagnosis = [diagnosis]
        if isinstance(cluster_nums, int):
            cluster_nums = [cluster_nums]
        for (i, d, c) in set(self._clusters.keys()):
            if img_nums is not None and i not in img_nums:
                self._clusters.pop((i, d, c))
                continue
            if diagnosis is not None and d not in diagnosis:
                self._clusters.pop((i, d, c))
                continue
            if cluster_nums is not None and c not in cluster_nums:
                self._clusters.pop((i, d, c))

    def unfilter(self):
        self._clusters = self._clusters_backup.copy()
