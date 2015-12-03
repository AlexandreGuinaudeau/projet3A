import os
import pandas as pd
from configuration import CONFIG


xy_path = os.path.join(CONFIG.backup_data_path, "XY.csv")
pixtup_path = os.path.join(CONFIG.backup_data_path, "PixTup.csv")
npos_path = os.path.join(CONFIG.backup_data_path, "NPOS.csv")
out_path = os.path.join(CONFIG.backup_data_path, "out.csv")


def load_df(reload=False):
    if reload:
        pix_tup_df = pd.read_csv(pixtup_path, header=None)
        pix_tup_df = pix_tup_df.iloc[:, 0:17]
        xy_df = pd.read_csv(xy_path, header=None)
        npos_df = pd.read_csv(npos_path, header=None)
        concat_df = pd.concat([npos_df, xy_df, pix_tup_df], axis=1)
        concat_df.columns = ['img_num', 'x', 'y', 'diagnosis', 'M11', 'M12', 'M13', 'M14', 'M21', 'M22', 'M23', 'M24',
                             'M31', 'M32', 'M33', 'M34', 'M41', 'M42', 'M43', 'M44']
        concat_df.to_csv(out_path, index=False)
    else:
        concat_df = pd.read_csv(out_path)
    return concat_df


def get_clusters():
    """
    For each image, finds the clusters, saves them in files and adds their metadata to the metadata file
    """
    # TODO (Alexandre): Code

if __name__ == "__main__":
    df = load_df(reload=False)
    print(df)

