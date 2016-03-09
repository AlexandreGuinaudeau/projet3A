import os
import logging
from matplotlib import cm
logging.basicConfig(level=logging.DEBUG)


class _Config:
    def __init__(self):
        """
        Define general configuration:
        - Useful samples to be used in the tests
        - Useful Müller matrix elements Mij to be used in the tests
        - Parameters for machine learning algorithms

        Only complete the first part of the configuration.
        """
        # ################################# General configurations #################################
        # Set logging level (DEBUG, INFO, WARNING, ERROR or CRITICAL)
        self.logging_level = logging.DEBUG

        # Samples to apply machine learning algorithms on
        self.complete_samples = {4, 7, 10, 16, 17}
        self.all_samples = set(range(1, 18))
        # Maximum size of images (for visualization)
        self.height = 485
        self.width = 511

        # Columns to apply machine learning algorithms on (= Relevant columns)
        self.learning_Mij = ['M22', 'M23', 'M24', 'M32', 'M33', 'M34', 'M42', 'M43', 'M44']
        # Columns stored in database
        self.database_Mij = ['M22', 'M23', 'M24', 'M32', 'M33', 'M34', 'M42', 'M43', 'M44']

        # ################################# DO NOT MODIFY #################################
        # Paths
        root_path = os.path.realpath(os.path.join(os.path.realpath(__file__), ".."))
        self.db_path = os.path.join(root_path, "database")

        self.raw_data_path = os.path.join(self.db_path, "raw_data")
        self.raw_xy_path = os.path.join(self.raw_data_path, "XY.csv")
        self.raw_pixtup_path = os.path.join(self.raw_data_path, "PixTup.csv")
        self.raw_npos_path = os.path.join(self.raw_data_path, "NPOS.csv")

        self.backup_data_path = os.path.join(self.db_path, "backup_data")

        self.data_path = os.path.join(self.db_path, "data")
        self.model_path = os.path.join(self.data_path, "models")
        self.center_path = os.path.join(self.data_path, "centers.csv")
        self.current_data_path = os.path.join(self.db_path, "current_data.csv")
        self.preprocessed_data_path = os.path.join(self.db_path, "preprocessed_data.csv")

        self.visualization_path = os.path.join(root_path, "visualization_results")

        # See http://matplotlib.org/examples/color/colormaps_reference.html
        self.colormap = cm.gist_rainbow  # gist_rainbow, jet, rainbow, gist_earth...
        import pandas as pd
        self.mean = pd.Series({'M23': 0.003082, 'M24': -0.005716, 'M32': 0.012070,
                               'M34': -0.001741, 'M42': -0.012651, 'M43': 0.006854})
        self.std = pd.Series({'M23': 0.009970, 'M24': 0.020685, 'M32': 0.017238,
                              'M34': 0.026307, 'M42': 0.021539, 'M43': 0.024973})

    @property
    def relevant_columns(self):
        return ['img_num', 'diagnosis', 'cluster_num'] + self.learning_Mij

    @property
    def cluster_columns(self):
        return self.database_Mij + ['x', 'y', 'M11']

    @property
    def all_database_columns(self):
        return ['img_num', 'diagnosis', 'cluster_num'] + self.cluster_columns

    @property
    def all_mij_columns(self):
        return ['M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']

    @property
    def all_columns(self):
        return ['img_num', 'x', 'y', 'diagnosis', 'M11', 'M12', 'M13', 'M14',
                'M21', 'M22', 'M23', 'M24', 'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']

CONFIG = _Config()
