from utils.SimilarityEngines.sim_cosine import cosine_user


class UCBF():
    """
    Consider playlist as user, we first compute the cosine similarity between two users, according to the
    input user content matrix. Then, having the sim_matrix, together with the known user rating matrix,
    we can compute the estimate rating matrix for further prediction.
    """

    def __init__(self):

        self.sim_matrix = None
        self.e_rating = None

    def fit(self, urm, ucm_t, target_playlist, target_tracks, dataset, knn, shrink_factor):

        # initialization
        pl_id_list = list(target_playlist)
        tr_id_list = list(target_tracks)

        # compute Similarity matrix
        target_list = [dataset.map_id_index_pl(x) for x in pl_id_list]
        sim_matrix = cosine_user(ucm_t, target_list, shrink_factor, knn)
        self.sim_matrix = sim_matrix

        # eliminate non related tracks
        urm_cleaned = urm.tocsc()[:, [dataset.map_id_index_tr(x)
                                      for x in tr_id_list]].tocsr()
        urm_cleaned_t = urm_cleaned.transpose().tocsr()

        # compute ratings
        e_rating_t = urm_cleaned_t.dot((sim_matrix.transpose()).tocsr())
        e_rating = e_rating_t.transpose().tocsr()

        # apply mask for eliminating already rated items
        urm_mask = urm_cleaned[[dataset.map_id_index_pl(x)
                                for x in pl_id_list]]
        e_rating[urm_mask.nonzero()] = 0
        e_rating.eliminate_zeros()
        self.e_rating = e_rating

    def get_estimate_rating(self):
        return self.e_rating

    def get_sim_matirx(self):
        return self.sim_matrix
