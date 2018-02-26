from scipy.spatial import distance_matrix
from scipy.sparse import *
import csv
from utils.csv_operations import load_column_float


def build_ual(dataset, attr, weight):
    """
    Build a playlist attribute dict according to './data/playlists_final.csv'
    :param dataset:
    :param attr: indicate the attribute to be used
    `           [numtracks, duration, created_at]
    :param weight: indicate the weight of each attr
    :return: ual: list of list
                  idx: playlist_idx
                  value: list of attr values
                    [numtracks, duration, created_at] scale (0, 1)
    """
    path = './data/playlists_final.csv'

    # find maximum value of attributes
    if attr[0] != 0:
        numtracks = load_column_float(path, 'numtracks')
        max_numtracks = max(numtracks) + 1
    if attr[1] != 0:
        duration = load_column_float(path, 'duration')
        max_duration = max(duration) + 1
    if attr[2] != 0:
        created_at = load_column_float(path, 'created_at')
        max_created_at = max(created_at) + 1

    ual = list(range(0, dataset.num_pl))

    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            index_pl = dataset.mapper_id_index_pl[row['playlist_id']]
            attr_list = [0, 0, 0]

            # numtracks
            if attr[0] != 0:
                value_numtracks = float(row['numtracks']) / max_numtracks
                attr_list[0] = value_numtracks * weight[0]

            # duration
            if attr[1] != 0:
                value_duration = float(row['duration']) / max_duration
                attr_list[1] = value_duration * weight[1]

            # created_at
            if attr[2] != 0:
                value_created_at = float(row['created_at']) / max_created_at
                attr_list[2] = value_created_at * weight[2]

            ual[index_pl] = attr_list

    return ual

