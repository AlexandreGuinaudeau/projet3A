import os
import matplotlib.pyplot as plt

from configuration import CONFIG


class Visualization:
    def __init__(self, cdb, min_value=-3, max_value=3):
        self.cdb = cdb
        self.min_value = min_value
        self.max_value = max_value

    def mij_histogram(self, i, j, out_path=None):
        centers3 = self.cdb.filter(diagnosis=3)['M%i%i' % (i, j)].tolist()
        centers4 = self.cdb.filter(diagnosis=4)['M%i%i' % (i, j)].tolist()
        plt.hist(centers3, histtype='step', facecolor='red')
        plt.hist(centers4, histtype='step', facecolor='green')
        plt.title('M%i%i' % (i, j))
        # plt.axis([self.min_value, self.max_value, 0, 1])
        plt.grid(True)
        if out_path is None:
            out_path = os.path.join(CONFIG.visualization_path, "Histogrammes", "M%i%iHistogram.png" % (i, j))
        plt.savefig(out_path)
        plt.clf()
