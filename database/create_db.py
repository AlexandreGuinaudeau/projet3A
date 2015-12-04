import os
import sys
import pandas as pd
from sklearn.cluster import KMeans

from configuration import CONFIG


xy_path = os.path.join(CONFIG.backup_data_path, "XY.csv")
pixtup_path = os.path.join(CONFIG.backup_data_path, "PixTup.csv")
npos_path = os.path.join(CONFIG.backup_data_path, "NPOS.csv")
temp_path = os.path.join(CONFIG.backup_data_path, "temp.csv")


def load_df(reload=False):
    if reload:
        pix_tup_df = pd.read_csv(pixtup_path, header=None)
        pix_tup_df = pix_tup_df.iloc[:, 0:17]
        xy_df = pd.read_csv(xy_path, header=None)
        npos_df = pd.read_csv(npos_path, header=None)
        concat_df = pd.concat([npos_df, xy_df, pix_tup_df], axis=1)
        concat_df.columns = ['img_num', 'x', 'y', 'diagnosis', 'M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                             'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']
        concat_df.to_csv(temp_path, index=False)
    else:
        concat_df = pd.read_csv(temp_path)
    return concat_df


def delete_database():
    for file_name in os.listdir(CONFIG.data_path):
        os.remove(os.path.join(CONFIG.data_path, file_name))
    open(CONFIG.metadata_path, 'w').close()


def update_clusters(concat_df, img_nums=None, threshold=15):
    """
    Finds the clusters for each image and each diagnosis, saves them in csv files adn adds them to the metadata.

    Parameters
    ==========
    concat_df: pd.Dataframe, The concatenation of raw data.
    img_nums: array-like or int, Images to create in the database
    threshold: number, The maximum sum of distances between elements of clusters and their center.
    """
    delete_database()
    if img_nums is None:
        img_nums = CONFIG.all_samples
    elif isinstance(img_nums, int):
        img_nums = [img_nums]
    for img_num in img_nums:
        for d in [3, 4]:
            x = concat_df[concat_df['img_num'] == img_num][concat_df['diagnosis'] == d]
            distance_x = x[['M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24', 'M31', 'M32', 'M33', 'M34',
                            'M41', 'M42', 'M43', 'M44']]
            if len(distance_x):  # There are pixels in this image with this diagnosis.
                last = sys.maxsize
                inertia = sys.maxsize - threshold - 1
                k = 0
                while last - inertia >= threshold:
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
                    out_name = "cluster_%i_%i_%i.csv" % (img_num, cluster_num, d)
                    out_path = os.path.join(CONFIG.data_path, out_name)
                    df[df[0] == cluster_num][['x', 'y', 'M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24', 'M31',
                                              'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']]\
                        .to_csv(out_path, index=False)
                    with open(CONFIG.metadata_path, 'a') as in_f:
                        in_f.write("%i,%i,%i,%s\n" % (img_num, cluster_num, d, out_name))
        print("Created all clusters for image %i." % img_num)


if __name__ == "__main__":
    df = load_df(reload=False)
    import time
    start = time.time()
    update_clusters(df, img_nums=CONFIG.all_samples)
    print("Total time:", time.time() - start)
