import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

from configuration import CONFIG
from mueller_matrix import MuellerMatrix


class Visualization:
    def __init__(self, min_value=-3, max_value=3):
        self.mueller_matrices = {}
        for sample_nb in CONFIG.all_samples:
            self.mueller_matrices[sample_nb] = MuellerMatrix(sample_nb, logger_level=logging.DEBUG)
        self.min_value = min_value
        self.max_value = max_value

    def __getitem__(self, item):
        if item in self.mueller_matrices.keys():
            return self.mueller_matrices[item]
        raise KeyError(item)

    def mij_histogram(self, i, j, out_path=None, sample_nb=None):
        type3_l = self._get_mij_type_l(i, j, 3, sample_nb)
        type4_l = self._get_mij_type_l(i, j, 4, sample_nb)
        mini = min(min(type3_l), min(type4_l))
        maxi = max(max(type3_l), max(type4_l))
        plt.hist(type3_l, bins=100, range=(mini, maxi), histtype='step', facecolor='red')
        plt.hist(type4_l, bins=100, range=(mini, maxi), histtype='step', facecolor='green')
        plt.title('M%i%i' % (i, j))
        # plt.axis([self.min_value, self.max_value, 0, 1])
        plt.grid(True)
        if out_path is None:
            out_path = os.path.join(CONFIG.visualization_path, "M%i%iHistogram.png" % (i, j))
        plt.savefig(out_path)
        plt.clf()

    def _get_mij_type_l(self, i, j, area_type, sample_nb):
        pix_tup_df = pd.read_csv(CONFIG.pixtup_path, header=None)
        mij_type_s = pd.Series()
        if sample_nb is None:
            for sample_nb in self.mueller_matrices.keys():
                mat_pix_tup_df = pix_tup_df.iloc[CONFIG.npos_index[sample_nb]:CONFIG.npos_index[sample_nb+1]]
                mat_pix_tup_df = mat_pix_tup_df[mat_pix_tup_df[0] == area_type]
                mij_type_s = mij_type_s.append(mat_pix_tup_df[4*(i-1) + j])
        else:
            mat_pix_tup_df = pix_tup_df.iloc[CONFIG.npos_index[sample_nb]:CONFIG.npos_index[sample_nb+1]]
            mat_pix_tup_df = mat_pix_tup_df[mat_pix_tup_df[0] == area_type]
            mij_type_s = mij_type_s.append(mat_pix_tup_df[4*(i-1) + j])
        return list(mij_type_s)

    def get_cluster_center(self, sample_nb):
        pix_tup_df = pd.read_csv(CONFIG.pixtup_path, header=None)
        pix_tup_df = pix_tup_df.iloc[CONFIG.npos_index[sample_nb]:CONFIG.npos_index[sample_nb+1]]
        pix_tup_df = pix_tup_df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
        print(pix_tup_df.mean())
        # sample_dir = self.mueller_matrices[sample_nb].sample_dir
        # for file_name in os.listdir(os.path.join(sample_dir, "metadata")):
        #     if file_name.startswith("XY"):
        #         xy_df = pd.read_csv(os.path.join(sample_dir, "metadata", file_name), header=None)
        #         for index, row in xy_df.iterrows():


if __name__ == "__main__":
    v = Visualization()
    # v.get_cluster_center(1)
    for i in [2, 3, 4]:
        for j in {2, 3, 4}.difference({i}):
            v.mij_histogram(i, j, None, 16)
            print("Done %i %i" % (i, j))

