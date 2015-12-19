import os
from matplotlib import cm


class _Config:
    def __init__(self):
        # All samples
        self.complete_samples = {4, 7, 10, 16, 17}
        self.all_samples = set(range(1, 18))

        # Database
        root_path = os.path.realpath(os.path.join(os.path.realpath(__file__), ".."))
        self.visualization_path = os.path.join(root_path, "visualization_results")
        self.db_path = os.path.join(root_path, "database")
        self.backup_data_path = os.path.join(self.db_path, "backup_data")
        self.data_path = os.path.join(self.db_path, "data")
        self.metadata_path = os.path.join(self.data_path, "metadata.csv")
        self.center_path = os.path.join(self.data_path, "centers.csv")
        self.variances_path = os.path.join(self.data_path, "variances.csv")
        self.is_normed_path = os.path.join(self.data_path, ".is_normed")

        # Columns
        # Columns to apply machine learning algorithms on (= Relevant columns)
        self.learning_columns = ['M23', 'M24', 'M32', 'M34', 'M42', 'M43']
        self.database_columns = ['M22', 'M23', 'M24', 'M32', 'M33', 'M34', 'M42', 'M43', 'M44']
        # ['M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24', 'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']

        # See http://matplotlib.org/examples/color/colormaps_reference.html
        self.colormap = cm.gist_rainbow  # gist_rainbow, jet, rainbow, gist_earth...

    @property
    def relevant_columns(self):
        return ['img_num', 'diagnosis', 'cluster_num'] + self.learning_columns

    @property
    def cluster_columns(self):
        return self.database_columns + ['x', 'y', 'M11']

    @property
    def all_columns(self):
        return ['img_num', 'diagnosis', 'cluster_num'] + self.cluster_columns

CONFIG = _Config()
