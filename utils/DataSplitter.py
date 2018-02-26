from tqdm import tqdm
from utils import pickle_operations as pickle_ops
from utils import lil_operations as lil_ops
import math
import random


class Splitter():
    """
    A Splitter is used to split the given Dataset
    """

    def __init__(self):
        # split data into five folds
        self.folds = 5
        # minimal number of tracks in a sample playlist
        self.min_playlist_size = 10
        # expect size of test set
        self.test_size = 100

    def cross_validation(self, interaction):
        # initialize
        # a list of sample dict(playlist: [five tracks]) for each fold
        sample_list = []
        # a list of list of target tracks for each fold
        target_list = []
        for i in range(0, self.folds):
            sample_list.append({})
            target_list.append(list())

        # list of feasible playlist
        f_playlists = [x for x in interaction.keys()
                       if len(interaction[x]) >= self.min_playlist_size]
        # number of feasible playlist
        num_f_pls = len(f_playlists)

        fold_size = math.ceil(num_f_pls / self.folds)
        # shuffle the list of playlist
        random.shuffle(f_playlists)

        # divide feasible playlist to each fold
        # a list of playlist list
        fold_playlists = [f_playlists[x:x + fold_size]
                          for x in range(0, len(f_playlists), fold_size)]

        # sample five tracks from each feasible playlist
        num_sample = 5
        fold_idx = 0
        for fold_pl in fold_playlists:
            fold_tg_trs = set()
            for pl in fold_pl:
                fold_sample_trs = random.sample(interaction[pl], num_sample)
                sample_list[fold_idx][pl] = fold_sample_trs
                fold_tg_trs = fold_tg_trs.union(fold_sample_trs)
            target_list[fold_idx] = fold_tg_trs
            fold_idx += 1

        return sample_list, target_list

    def build_testset(self, ds, surname):
        path = './test_data/'
        for i in tqdm(range(0, self.test_size)):
            name_count = i * self.folds
            interaction = ds.get_interaction()
            sample_list, target_list = self.cross_validation(interaction)

            for k in range(0, self.folds):
                idx = name_count + k
                name = str(idx).zfill(5)
                name = surname + name + ".txt"
                itr_file = path + "itr" + name
                urm_file = path + "urm" + name
                ev_file = path + "ev" + name
                tg_file = path + "tg" + name

                urm = ds.build_urm()
                interaction = ds.get_interaction()
                for pl in sample_list[k].keys():
                    pl_idx = ds.map_id_index_pl(pl)
                    for tr in sample_list[k][pl]:
                        tr_idx = ds.map_id_index_tr(tr)
                        urm[pl_idx, tr_idx] = 0
                        interaction[pl].remove(tr)
                target_tracks = target_list[k]
                evaluation = sample_list[k]

                pickle_ops.save(itr_file, interaction)
                lil_ops.save_lil(urm_file, urm)
                pickle_ops.save(ev_file, evaluation)
                pickle_ops.save(tg_file, target_tracks)




