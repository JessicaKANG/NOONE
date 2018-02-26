from scipy.sparse import *
import csv
import math
from utils import csv_operations as csv_opt
from utils.csv_operations import load_column_float


def build_ucm(dataset, attr, weight, GROUP):
    """
    Build an user content matrix according to './data/playlists_final.csv'
    :param attr: a list of tag to indicate which attribute to be used
                    [owner, title, numtracks, duration, created_at]
    :param weight: a list of weight indicates the weight of each attr
    :return: ucm
    """
    # group together similar size numtracks
    #GROUP_nt = 50
    interval_nt = 1
    interval_dur = 1
    interval_ct = 1

    GROUP_nt = GROUP[0]
    GROUP_dur = GROUP[1]
    GROUP_ct = GROUP[2]

    path = './data/playlists_final.csv'

    if attr[2] != 0:
        numtracks = load_column_float(path, 'numtracks')
        max_numtracks = max(numtracks)
        min_numtracks = min(numtracks)
        interval_nt = (max_numtracks - min_numtracks) / GROUP_nt
        print("numtracks: max: ", max_numtracks, " min: ", min_numtracks, " interval: ", interval_nt, " num: ", len(numtracks))

    if attr[3] != 0:
        duration = load_column_float(path, 'duration')
        max_duration = max(duration)
        min_duration = min(duration)
        interval_dur = (max_duration - min_duration) / GROUP_dur
        print("duration: max: ", max_duration, " min: ", min_duration, " interval: ", interval_dur, " num: ", len(duration))

    if attr[4] != 0:
        created_at = load_column_float(path, 'created_at')
        max_created_at = max(created_at)
        min_created_at = min(created_at)
        interval_ct = (max_created_at - min_created_at) / GROUP_ct
        print("created_at: max: ", max_created_at, " min: ", min_created_at, " interval: ", interval_ct, " num: ", len(created_at))



    m_owner = lil_matrix((dataset.num_attr_pl['owner'], dataset.num_pl))
    m_title = lil_matrix((dataset.num_attr_pl['title'], dataset.num_pl))

    m_numtracks = lil_matrix((GROUP_nt, dataset.num_pl))
    m_duration = lil_matrix((GROUP_dur, dataset.num_pl))
    m_created_at = lil_matrix((GROUP_ct, dataset.num_pl))
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            index_pl = dataset.mapper_id_index_pl[row['playlist_id']]
            # owner
            if attr[0] != 0:
                index_ow = dataset.mapper_attr_pl['owner'][row['owner']]
                m_owner[index_ow, index_pl] = weight[0]
            # title
            if attr[1] != 0:
                titles = csv_opt.parse_list(row['title'])
                if len(titles):
                    title_set = set()
                    title_freq = {}
                    title_num = len(titles)
                    tf_idf = {}
                    for title in titles:
                        title_set.add(title)
                        count = title_freq.setdefault(title, 0)
                        title_freq[title] = count + 1
                    # tf-idf
                    for title in title_set:
                        tf_idf[title] = (title_freq[title] / title_num) * math.log(
                            dataset.num_pl / (dataset.corpus_title[title] + 1))
                        index_tt = dataset.mapper_attr_pl['title'][title]
                        m_title[index_tt, index_pl] = tf_idf[title] * weight[1]
            # numtracks
            if attr[2] != 0:
                f = row['numtracks']
                if f != '':
                    index_nt = math.floor((float(row['numtracks']) - min_numtracks) / interval_nt)
                    if index_nt == GROUP_nt:
                        index_nt -= 1
                    m_numtracks[index_nt, index_pl] = weight[2]

            # duration
            if attr[3] != 0:
                f = row['duration']
                if f != '':
                    index_dr = math.floor((float(row['duration']) - min_duration) / interval_dur)
                    if index_dr == GROUP_dur:
                        index_dr -= 1
                    m_duration[index_dr, index_pl] = weight[3]

            # created_at
            if attr[4] != 0:
                f = row['created_at']
                if f != '':
                    index_ct = math.floor((float(row['created_at']) - min_created_at) / interval_ct)
                    if index_ct == GROUP_ct:
                        index_ct -= 1
                    m_created_at[index_ct, index_pl] = weight[3]

    ucm = lil_matrix((0, dataset.num_pl))
    if attr[0] != 0:
        ucm = vstack([ucm, m_owner])
    if attr[1] != 0:
        ucm = vstack([ucm, m_title])
    if attr[2] != 0:
        ucm = vstack([ucm, m_numtracks])
    if attr[3] != 0:
        ucm = vstack([ucm, m_duration])
    if attr[4] != 0:
        ucm = vstack([ucm, m_created_at])
    return ucm