import os
from matplotlib import cm


class _Config:
    def __init__(self):
        # All samples
        self.complete_samples = {4, 7, 10, 16, 17}
        self.all_samples = set(range(1, 17))

        # Get Paths
        self.data_path = os.path.realpath(os.path.join(os.path.realpath(__name__), "..", "data"))
        self.in_sample_path = os.path.join(self.data_path, "sample_first_last.csv")
        self.xy_path = os.path.join(self.data_path, "XY.csv")
        self.pixtup_path = os.path.join(self.data_path, "PixTup.csv")

        self.npos_index = {1: 0, 2: 13787, 3: 38674, 4: 43118, 5: 51690, 6: 55047,
                           7: 55859, 8: 70072, 9: 93299, 10: 101594, 11: 107798, 12: 122664,
                           13: 129064, 14: 132474, 15: 137918, 16: 143697, 17: 146779, 18: 163041}

        self.visualization_path = os.path.join(self.data_path, "visualization")

        # See http://matplotlib.org/examples/color/colormaps_reference.html
        self.colormap = cm.gist_rainbow  # gist_rainbow, jet, rainbow, gist_earth...

CONFIG = _Config()
