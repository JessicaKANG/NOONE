from utils import csv_operations as csv_ops
from utils import lil_operations as lil_ops
from scipy.sparse import *
import os.path
import time
import copy


class Dataset():
    """
        A Dataset contains information captured from the given data:
        pls: total playlists
        trs: total tracks
        tg_tr: a list of target tracks from '../data/target_tracks.csv'
        tg_pl: a list of target playlists from '../data/target_playlists.csv'
        interaction: a dict with key: 'playlist_id'
                                 value: a list of track ids from '../data/train_final.csv'
        num_tr: number of total distinct tracks
        num_pl: number of total distinct playlists
        num_ow: number of total distinct owners
        num_attr_tr: a dict {'album': num_album, 'artist_id': num_artist_id, 'tags': num_tags,
                                'playcount': num_playcount, 'duration': num_duration}
        num_attr_pl: a dict

        mapper_id_index_tr
        mapper_index_id_tr
        mapper_attr_tr: a dict of dict {'album': {'album_id': album_index}, 'artist_id': {'artist_id': artist_index},
                                        'tags': {'tag_id': tag_index}, 'playcount': {'playcount': playcount_index},
                                        'duration': {'duration': duration_index}}
        mapper_id_tags: a dict key: track_id  value: tagset_local

        mapper_id_index_pl
        mapper_index_id_pl
        mapper_attr_pl;
        mapper_id_index_ow
        mapper_index_id_ow
        mapper_id_titles: a dict key: playlist_id value: titleset_local

        urm: a lil_matrix with playlist_index as row key and track_index as column key
                from '../data/train_final.csv

    """

    def __init__(self):
        self.pls = csv_ops.load_column('./data/playlists_final.csv', 'playlist_id')
        self.trs = csv_ops.load_column('./data/tracks_final.csv', 'track_id')
        self.tg_pl = csv_ops.load_column('./data/target_playlists.csv', 'playlist_id')
        self.tg_tr = csv_ops.load_column('./data/target_tracks.csv', 'track_id')
        self.interaction = csv_ops.load_interaction('./data/train_final.csv')

        self.urm = None

        self.mapper_id_id_tr_attr, self.mapper_id_index_tr, self.mapper_index_id_tr, self.mapper_attr_tr, \
            self.mapper_attr_index_tr, self.num_tr, self.num_attr_tr, \
            self.mapper_id_tags, self.corpus_tr = csv_ops.load_tracks('./data/tracks_final.csv')
        self.mapper_id_index_pl, self.mapper_index_id_pl, self.mapper_index_id_ow, self.mapper_id_index_pl_ow, \
            self.mapper_attr_pl, self.num_pl, self.num_attr_pl, self.mapper_id_titles, \
            self.corpus_title = csv_ops.load_playlists('./data/playlists_final.csv')

    def get_target_pl(self):
        return self.tg_pl.copy()

    def get_target_tr(self):
        return self.tg_tr.copy()

    def get_interaction(self):
        return copy.deepcopy(self.interaction)

    def map_id_index_tr(self, id_tr):
        return self.mapper_id_index_tr[id_tr]

    def map_id_index_pl(self, id_pl):
        return self.mapper_id_index_pl[id_pl]

    def map_id_index_ow(self, id_ow):
        return self.mapper_attr_pl['owner'][id_ow]

    def map_index_id_tr(self, index_tr):
        return self.mapper_index_id_tr[index_tr]

    def map_index_id_pl(self, index_pl):
        return self.mapper_id_index_pl[index_pl]

    def map_index_id_ow(self, index_ow):
        return self.mapper_index_id_ow[index_ow]

    def map_id_index_pl_ow(self, id_pl):
        return self.mapper_id_index_pl_ow[id_pl]

    def build_urm(self):
        """
        Build playlist-track rating matrix from interaction if haven't been built before,
        save and return the sparse matrix
        :return: playlist-track rating matrix
        """
        # check if it has been built
        if self.urm is None:
            # haven't built
            # check if it has been saved before
            path = './data/csr_urm.npz'
            if os.path.isfile(path):
                # saved before, load from file
                time_start = time.time()
                #print('[DataReader]: Loading urm from ' + path)
                self.urm = lil_ops.load_lil(path)
                time_dur = time.time() - time_start
                #print('[DataReader]: urm loaded with {:.2f} seconds'.format(time_dur))
                return self.urm
            # no saved, fresh build and save
            time_start = time.time()
            print('[DataReader]: Start building urm...')
            self.urm = lil_matrix((self.num_pl, self.num_tr))
            for pl_id, tr_ids in self.interaction.items():
                row = self.mapper_id_index_pl[pl_id]
                for tr_id in tr_ids:
                    column = self.mapper_id_index_tr[tr_id]
                    self.urm[row, column] = 1
            time_dur = time.time() - time_start
            print('[DataReader]: urm built with {:.2f} seconds'.format(time_dur))
            print('[DataReader]: Saving urm to ' + path)
            lil_ops.save_lil(path, self.urm)
        # already built
        return self.urm.copy()
