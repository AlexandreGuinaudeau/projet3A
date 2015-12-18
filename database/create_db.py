import os
import sys
import time
import pandas as pd
from sklearn.cluster import KMeans

from configuration import CONFIG


xy_path = os.path.join(CONFIG.backup_data_path, "XY.csv")
pixtup_path = os.path.join(CONFIG.backup_data_path, "PixTup.csv")
npos_path = os.path.join(CONFIG.backup_data_path, "NPOS.csv")
temp_path = os.path.join(CONFIG.backup_data_path, "temp.csv")


def load_df(reload=False):
    """
    Concatenates raw data into one Dataframe. Stores it in a temp file for more efficiency.
    """
    if reload or not os.path.isfile(temp_path):
        pix_tup_df = pd.read_csv(pixtup_path, header=None)
        pix_tup_df = pix_tup_df.iloc[:, 0:17]
        xy_df = pd.read_csv(xy_path, header=None)
        npos_df = pd.read_csv(npos_path, header=None)
        concat_df = pd.concat([npos_df, xy_df, pix_tup_df], axis=1)
        concat_df.columns = CONFIG.database_columns
        concat_df.to_csv(temp_path, index=False)
    else:
        concat_df = pd.read_csv(temp_path)
    return concat_df


def delete_database():
    """
    Removes all files from the database and initializes the metadata files.
    """
    for file_name in os.listdir(CONFIG.data_path):
        os.remove(os.path.join(CONFIG.data_path, file_name))
    open(CONFIG.metadata_path, 'w').close()
    open(CONFIG.center_path, 'w').close()
    open(CONFIG.variances_path, 'w').close()


def update_clusters(concat_df, img_nums, threshold, norm, save_centers):
    """
    Finds the clusters for each image and each diagnosis, saves them in csv files adn adds them to the metadata.
    Warning: Takes about 45 sec to compute.

    Parameters
    ==========
    concat_df: pd.Dataframe, the concatenation of raw data.
    img_nums: array-like or int, images to create in the database. Default: all images
    threshold: number, ratio between the last clustering and the following one
    norm: boolean, whether columns should be normed before clustering
    save_centers: boolean, whether to save the centers and variances of the clusters in a file to retrieve them faster.
    """
    delete_database()
    if norm:
        open(CONFIG.is_normed_path, 'w').close()
    if save_centers:
        with open(CONFIG.center_path, 'w') as in_f:
            in_f.write(",".join(CONFIG.all_columns)+"\n")
        with open(CONFIG.variances_path, 'w') as in_f:
            in_f.write(",".join(CONFIG.all_columns)+"\n")
    if img_nums is None:
        img_nums = CONFIG.all_samples
    elif isinstance(img_nums, int):
        img_nums = [img_nums]
    for img_num in img_nums:
        for d in [3, 4]:
            x = concat_df[concat_df['img_num'] == img_num][concat_df['diagnosis'] == d]
            if norm:
                for Mij in CONFIG.learning_columns:
                    x[Mij] = x['M11']*x[Mij]
            distance_x = x[CONFIG.learning_columns]
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
                    out_name = "cluster_%i_%i_%i.csv" % (img_num, d, cluster_num)
                    out_path = os.path.join(CONFIG.data_path, out_name)
                    cluster_df = df[df[0] == cluster_num-1]
                    cluster_df.to_csv(out_path, index=False)
                    with open(CONFIG.metadata_path, 'a') as in_f:
                        in_f.write("%i,%i,%i,%s\n" % (img_num, d, cluster_num, out_name))
                    if save_centers:
                        with open(CONFIG.center_path, 'a') as in_f:
                            center = cluster_df.mean()
                            line = "%i,%i,%i" % (img_num, d, cluster_num)
                            for column in CONFIG.cluster_columns:
                                line += "," + str(center[column])
                            in_f.write(line + "\n")
                        with open(CONFIG.variances_path, 'a') as in_f:
                            var = cluster_df.var()
                            line = "%i,%i,%i" % (img_num, d, cluster_num)
                            for column in CONFIG.cluster_columns:
                                line += "," + str(var[column])
                            in_f.write(line + "\n")
        print("Created all clusters for image %i." % img_num)


def create_database(img_nums=None, reload=False, threshold=4, norm=False, save_centers=True):
    df = load_df(reload=reload)
    start = time.time()
    update_clusters(df, img_nums=img_nums, threshold=threshold, norm=norm, save_centers=save_centers)
    print("Total time:", time.time() - start)


if __name__ == "__main__":
    create_database(norm=True)
