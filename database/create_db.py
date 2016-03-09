import os
import logging
import sys
import pandas as pd
from sklearn.cluster import KMeans

from configuration import CONFIG
from utils import load_df


logger = logging.getLogger(__name__)
logger.setLevel(CONFIG.logging_level)


def find_cluster_centers(img_nums, threshold, norm, normalize):
    """
    Finds the cluster centers for each image and each diagnosis.
    Warning: Takes about 45 sec to compute.

    Parameters
    ==========
    img_nums: array-like or int, images to create in the database. Default: all images
    threshold: number, ratio between the last clustering and the following one
    norm: boolean, whether columns should be normed before clustering
    normalize: boolean, whether columns should be normalized (variance=1) before clustering
    """
    logger.debug('Finding cluster centers...')
    open(CONFIG.center_path, 'w').close()
    with open(CONFIG.center_path, 'w') as in_f:
        in_f.write(",".join(CONFIG.all_database_columns) + "\n")
    if img_nums is None:
        img_nums = CONFIG.all_samples
    elif isinstance(img_nums, int):
        img_nums = [img_nums]

    current_df = load_df(reload=False)
    standard_deviations = current_df.std()
    for img_num in img_nums:
        for d in [3, 4]:
            x = current_df[current_df['img_num'] == img_num][current_df['diagnosis'] == d]
            if norm:
                for Mij in CONFIG.learning_Mij:
                    x[Mij] = x['M11']*x[Mij]
            if normalize:
                for Mij in CONFIG.learning_Mij:
                    x[Mij] = x[Mij]/standard_deviations[Mij]
            distance_x = x[CONFIG.learning_Mij]
            if len(distance_x):  # There are pixels in this image with this diagnosis.
                last = sys.maxsize
                inertia = sys.maxsize/2
                k = 0
                while last - inertia >= last/threshold:
                    k += 1
                    last = inertia
                    km = KMeans(k)
                    km.fit(distance_x)
                    inertia = km.inertia_
                k -= 1
                km = KMeans(k)
                km.fit(distance_x)
                df = pd.DataFrame(pd.concat([x.reset_index(), pd.Series(km.labels_)], axis=1))
                for cluster_num in range(1, k+1):
                    cluster_df = df[df[0] == cluster_num-1]
                    with open(CONFIG.center_path, 'a') as in_f:
                        center = cluster_df.mean()
                        line = "%i,%i,%i" % (img_num, d, cluster_num)
                        for column in CONFIG.cluster_columns:
                            line += "," + str(center[column])
                        in_f.write(line + "\n")
        logger.info("Created all cluster centers for image %i." % img_num)
