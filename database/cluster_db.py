import os
import pandas as pd
from configuration import CONFIG


class Cluster:
    """
    Represents a group of 'similar' points.
    Call norm() to multiply all Mij by M11.

    Attributes
    ==========

    img_num: int >=1, the number of the image the points come from
    cluster_num: int >=1, an id, unique among the clusters of this image.
        There is no hierarchy among clusters of an image.
    diagnosis: 3 or 4, the diagnosis of all the points of the cluster.
    file_path: str, the path to the csv file storing the data of the points from this cluster.
    df: pd.Dataframe, the Mij values of the points. It is loaded when called for the first time.
    center: pd.Series, the Mij values of the center of the cluster (mean value of each Mij)
    var: pd.Series, for each Mij, the variance of the points of the cluster along this axis.
    """
    def __init__(self, img_num, cluster_num, diagnosis, file_path, center=None, variances=None):
        self.img_num = img_num
        self.cluster_num = cluster_num
        self.diagnosis = diagnosis
        self.file_path = file_path
        self._df = None
        self._center = center
        self._variances = variances
        self._normed = os.path.isfile(CONFIG.is_normed_path)

    def __str__(self):
        return "<Cluster (%i, %i): Diagnosis=%i>" % (self.img_num, self.cluster_num, self.diagnosis)

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


class Image:
    """
    Represents the points of an image.

    Attributes
    ==========

    img_num: int >=1, the id of the image.
    clusters: dict, maps ids of clusters of this image to the cluster.
    """
    def __init__(self, img_num):
        self.img_num = img_num
        self.clusters = {}

    def __str__(self):
        return "<Image %i: %i clusters>" % (self.img_num, self.nb_clusters)

    @property
    def nb_clusters(self):
        return len(self.clusters)

    def add_cluster(self, cluster_num, cluster):
        self.clusters[cluster_num] = cluster

    def __getitem__(self, item):
        return self.clusters[item]

    def save(self):
        # TODO (Pierre): code
        """ Saves the image (with its clusters) as png. """


class ClusterDB:
    """
    Represents all the data.
    To access only centers of clusters or variances, use cdb.centers and cdb.variances.

    Examples
    ========

    cdb = ClusterDB(CONFIG.db_path, CONFIG.metadata_path)
    for img in cdb.images:
        ...
    for cluster in cdb.clusters:
        ...
    img1 = cdb[1]
    cluster1_2 = cdb[1, 2]
    """
    def __init__(self, db_path, metadata_path, center_path="", variances_path=""):
        self._db_path = db_path
        self._images = {}
        self._clusters = {}
        self.centers = pd.read_csv(center_path) if os.path.isfile(center_path) else None
        self.variances = pd.read_csv(variances_path) if os.path.isfile(variances_path) else None
        self._load_metadata(metadata_path)

    def __str__(self):
        return "<ClusterDB: %i Images, %i Clusters>" % (len(self._images), len(self._clusters))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._images[item]
        if isinstance(item, tuple):
            return self._clusters[item]
        raise KeyError(item)

    @property
    def images(self):
        return self._images.values()

    @property
    def clusters(self):
        return self._clusters.values()

    def _load_metadata(self, metadata_path):
        df = pd.read_csv(metadata_path, header=None,
                         names=['img_num', 'cluster_num', 'diagnosis', 'file_name'])
        for index, row in df.iterrows():
            img_num, cluster_num, diagnosis, file_name = row
            file_path = os.path.join(self._db_path, file_name)
            center = None
            if self.centers is not None:
                center = self.centers[self.centers['img_num'] == img_num][self.centers['cluster_num'] == cluster_num]
                center = pd.Series(center[center[diagnosis] == diagnosis])
            self._clusters[(img_num, cluster_num)] = \
                Cluster(img_num, cluster_num, diagnosis, file_path, center=center)
            if img_num not in self._images.keys():
                self._images[img_num] = Image(img_num)
            self._images[img_num].add_cluster(cluster_num, self._clusters[(img_num, cluster_num)])
