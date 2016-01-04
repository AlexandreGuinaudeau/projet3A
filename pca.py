import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from database import ClusterDB
from configuration import CONFIG

class Princ_Comp :

    #doc http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
    def __init__(self, with_outliers = False):
        self.cdb = ClusterDB()
        self.pca = PCA(whiten=True)

        if not with_outliers:
            self.centers = self.cdb.centers.drop([22, 35])

        self.data_train = self.centers[CONFIG.pca_col]
        self.data_label = self.centers['diagnosis']
        self.pca.fit(self.data_train)


    def plot_new_points(self):
        X_r = self.pca.transform(self.data_train)
        plt.scatter(X_r[:, 0], X_r[:, 1], c=['b' if x == 3 else 'r' for x in self.data_label])
        plt.show()

    def plot_first_comp(self):
        index = np.arange(len(self.data_train.columns.values))
        bar_width = 0.35

        plt.bar(index, self.pca.components_[1], bar_width)
        label = self.data_train.columns.values
        plt.xticks(index + bar_width, label)
        plt.legend()
        plt.tight_layout()

        plt.show()

    def plot_explained_variance_ratio(self):
        plt.plot(self.pca.explained_variance_ratio_)
        plt.show()
            #doc http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

if __name__ == "__main__":
    pca = Princ_Comp()
    pca.plot_new_points()
    pca.plot_first_comp()
    pca.plot_explained_variance_ratio()