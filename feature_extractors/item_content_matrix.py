from scipy.sparse import *
import csv
import math
from utils import csv_operations as csv_opt


def build_icm(dataset, attr, weight):
    """
    Build an item content matrix according to './data/tracks_final.csv'
    :param attr: a list of tag to indicate which attribute to be used
                    [album, artist_id, tags, playcount, duration]
    :param weight: a list of weight indicates the weight of each attr
    :return: icm: row: track attribute
                  column: track
    """
    path = './data/tracks_final.csv'
    m_album = lil_matrix((dataset.num_attr_tr['album'], dataset.num_tr))
    m_artist_id = lil_matrix((dataset.num_attr_tr['artist_id'], dataset.num_tr))
    m_tags = lil_matrix((dataset.num_attr_tr['tags'], dataset.num_tr))
    m_playcount = lil_matrix((dataset.num_attr_tr['playcount'], dataset.num_tr))
    m_duration = lil_matrix((dataset.num_attr_tr['duration'], dataset.num_tr))
    with open(path, newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter='\t')
        for row in reader:
            index_tr = dataset.mapper_id_index_tr[row['track_id']]
            # album
            if attr[0] != 0:
                albums = csv_opt.parse_list(row['album'])
                for album in albums:
                    index_ab = dataset.mapper_attr_tr['album'][album]
                    m_album[index_ab, index_tr] = weight[0]
            # artist_id
            if attr[1] != 0:
                index_at = dataset.mapper_attr_tr['artist_id'][row['artist_id']]
                m_artist_id[index_at, index_tr] = weight[1]
            # tags
            if attr[2] != 0:
                tags = csv_opt.parse_list(row['tags'])
                if len(tags):
                    tag_set = set()
                    tag_freq = {}
                    tag_num = len(tags)
                    tf_idf = {}
                    for tag in tags:
                        tag_set.add(tag)
                        count = tag_freq.setdefault(tag, 0)
                        tag_freq[tag] = count + 1
                    # tf-idf
                    for tag in tag_set:
                        tf_idf[tag] = (tag_freq[tag] / tag_num) * math.log(
                            dataset.num_tr / (dataset.corpus_tr['tags'][tag] + 1))
                        index_tg = dataset.mapper_attr_tr['tags'][tag]
                        m_tags[index_tg, index_tr] = tf_idf[tag] * weight[2]

            # playcount
            if attr[3] != 0:
                if row['playcount'] is not None and row['playcount'] != '':
                    index_pc = dataset.mapper_attr_tr['playcount'][float(row['playcount'])]
                    m_playcount[index_pc, index_tr] = weight[3]
            # duration
            if attr[4] != 0:
                duration = float(row['duration'])
                if duration != -1:
                    index_dr = dataset.mapper_attr_tr['duration'][duration]
                    m_duration[index_dr, index_tr] = weight[4]
    icm = lil_matrix((0, dataset.num_tr))
    if attr[0] != 0:
        icm = vstack([icm, m_album])
    if attr[1] != 0:
        icm = vstack([icm, m_artist_id])
    if attr[2] != 0:
        icm = vstack([icm, m_tags])
    if attr[3] != 0:
        icm = vstack([icm, m_playcount])
    if attr[4] != 0:
        icm = vstack([icm, m_duration])

    return icm