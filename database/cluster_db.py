import os
import pandas as pd


class Cluster:
    def __init__(self, img_num, cluster_num, diagnosis, file_path):
        self.img_num = img_num
        self.cluster_num = cluster_num
        self.diagnosis = diagnosis
        self.file_path = file_path
        self._df = None
        self._normed = False

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

    @property
    def center(self):
        return self.df.mean()

    @property
    def variances(self):
        return self.df.var()

    def _load_df(self):
        self._df = pd.read_csv(self.file_path)[['M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                                                'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']]

    def save(self):
        # TODO (Pierre): code
        """ Saves the cluster as png. """


class Image:
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
    def __init__(self, db_path, metadata_path):
        self._db_path = db_path
        self._images = {}
        self._clusters = {}
        self._load_metadata(metadata_path)

    def __str__(self):
        return "<ClusterDB: %i Images, %i Clusters>" % (len(self._images), len(self._clusters))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._images[item]
        if isinstance(item, tuple):
            return self._clusters[item]
        raise KeyError(item)

    def _load_metadata(self, metadata_path):
        df = pd.read_csv(metadata_path, header=None,
                         names=['img_num', 'cluster_img', 'diagnosis', 'file_name'])
        for index, row in df.iterrows():
            img_num, cluster_num, diagnosis, file_name = row
            file_path = os.path.join(self._db_path, file_name)
            self._clusters[(img_num, cluster_num)] = \
                Cluster(img_num, cluster_num, diagnosis, file_path)
            if img_num not in self._images.keys():
                self._images[img_num] = Image(img_num)
            self._images[img_num].add_cluster(cluster_num, self._clusters[(img_num, cluster_num)])
