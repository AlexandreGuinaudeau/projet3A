import os
from matplotlib import cm


class _Config:
    def __init__(self):
        # All samples
        self.complete_samples = {4, 7, 10, 16, 17}
        self.all_samples = set(range(1, 17))

        # Get Paths
        self.db_path = os.path.realpath(os.path.join(os.path.realpath(__file__), "..", "database"))
        self.backup_data_path = os.path.join(self.db_path, "backup_data")
        self.data_path = os.path.join(self.db_path, "data")

        self.visualization_path = os.path.join(self.data_path, "visualization")

        # See http://matplotlib.org/examples/color/colormaps_reference.html
        self.colormap = cm.gist_rainbow  # gist_rainbow, jet, rainbow, gist_earth...

CONFIG = _Config()
