import numpy as np


def predict(R_hat, target_playlist, target_tracks, at=5):

    pl_id_list = list(target_playlist)
    tr_id_list = list(target_tracks)
    recs = {}
    for i in range(0, R_hat.shape[0]):
        pl_id = pl_id_list[i]
        pl_row = R_hat.data[R_hat.indptr[i]:
                                 R_hat.indptr[i + 1]]
        # get top 5 indeces. argsort, flip and get first at-1 items
        sorted_row_idx = np.flip(pl_row.argsort(), axis=0)[0:at]
        track_cols = [R_hat.indices[R_hat.indptr[i] + x]
                      for x in sorted_row_idx]
        tracks_ids = [tr_id_list[x] for x in track_cols]
        recs[pl_id] = tracks_ids
    return recs