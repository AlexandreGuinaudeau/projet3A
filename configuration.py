import os


class _Config:
    def __init__(self, sample_nb=17):
        # Set sample nb
        self.sample_nb = sample_nb

        # Get Paths
        self.data_path = os.path.join(os.path.realpath(__name__), "..", "data", "Sample%i" % sample_nb)
        self.in_sample_path = os.path.join(self.data_path, "sample_first_last.csv")

        self.xy_path = os.path.join(os.path.realpath(__name__), "..", "data", "XY.csv")
        self.raw_pixtup_path = os.path.join(os.path.realpath(__name__), "..", "data", "PixTup.csv")
        self.data_pixtup_path = os.path.join(self.data_path, "PixTup%i.csv" % sample_nb)

        self.npos_index = {1: 0, 2: 13787, 3: 38674, 4: 43118, 5: 51690, 6: 55047,
                           7: 55859, 8: 70072, 9: 93299, 10: 101594, 11: 107798, 12: 122664,
                           13: 129064, 14: 132474, 15: 137918, 16: 143697, 17: 146779, 18: 163041}

CONFIG = _Config()
