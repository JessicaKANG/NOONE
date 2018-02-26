class Evaluator():
    """
    Given the recommendation result and real answer, compute
    the mean average precision (map@5);
    """
    def evaluate_ent(self, recs, answer):
        """
        Evaluation for entire target playlists
        :param recs: dict key: pl_id value:[five tr_ids]
        :param answer: dict key: pl_id  value: [five tr_ids]
        :return: map@5
        """
        mean_ap = 0
        num_pls = 0

        for pl in recs.keys():
            num_pls += 1
            recs_ind = recs[pl]
            ans_ind = answer[pl]
            ap = self.evaluate_ind(recs_ind, ans_ind)

            mean_ap += ap
        mean_ap = mean_ap / num_pls

        return mean_ap

    def evaluate_ind(self, recs, answer):
        """
        Individually evaluate one playlist
        :param recs: list [five tr_ids]
        :param answer: list [five tr_ids]
        :return: ap@5
        """
        # initialize
        num_relevant = 0
        rank = 0
        avg_precision = 0

        for tr in recs:
            rank += 1
            if tr in answer:
                num_relevant += 1
                precision = num_relevant / rank
                avg_precision += precision
        if num_relevant:
            avg_precision = avg_precision / num_relevant
        else:
            avg_precision = 0

        return avg_precision