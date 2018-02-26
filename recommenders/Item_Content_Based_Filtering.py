from utils.SimilarityEngines.sim_cosine import cosine_item


class ICBF():
    """
    Consider track as item, we first compute the cosine similarity between two items, according to the
    input item content matrix. Then, having the sim_matrix, together with the known user rating matrix,
    we can compute the estimate rating matrix for further prediction.
    """

    def __init__(self):

        self.sim_matrix = None
        self.e_rating = None

    def fit(self, urm, icm, target_playlist, target_tracks, dataset, knn, shrink_factor):

        # initialization
        pl_id_list = list(target_playlist)
        tr_id_list = list(target_tracks)

        # compute Similarity matrix
        target_list = [dataset.map_id_index_tr(x) for x in tr_id_list]
        sim_matrix = cosine_item(icm, target_list, shrink_factor, knn)
        self.sim_matrix = sim_matrix

        # eliminate non related playlist
        urm_cleaned = urm.tocsr()[[dataset.map_id_index_pl(x)
                                   for x in pl_id_list]]

        # compute ratings
        e_rating = urm_cleaned.dot((sim_matrix.transpose()).tocsr())

        # apply mask for eliminating already rated items
        urm_mask = urm_cleaned.tocsc()[:, [dataset.map_id_index_tr(x)
                                           for x in tr_id_list]].tocsr()
        e_rating[urm_mask.nonzero()] = 0
        e_rating.eliminate_zeros()
        self.e_rating = e_rating

    def get_estimate_rating(self):
        return self.e_rating

    def get_sim_matrix(self):
        return self.sim_matrix

