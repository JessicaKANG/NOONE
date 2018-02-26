from recommenders.User_Based_Collaborative_Filtering import UCF
from recommenders.User_Content_Based_Filtering import UCBF
from recommenders.Item_Based_Collaborative_Filtering import ICF
from recommenders.Item_Content_Based_Filtering import ICBF

from feature_extractors.user_content_matrix import build_ucm
from feature_extractors.item_content_matrix import build_icm

from utils.predictor import predict
from utils.csv_operations import write_submission

from scipy.sparse import *


class HYBRID ():

    def __init__(self):

        self.eR_ucf = None
        self.eR_ucbf = None
        self.eR_icf = None
        self.eR_icbf = None

        self.e_rating = None

        self.target_playlist = []
        self.target_tracks = []

    def fit(self, urm, target_playlist, target_tracks, ds, coefficient):
        """
        :param urm:
        :param target_playlist:
        :param target_tracks:
        :param ds:
        :param coefficient: [ucbf, ucf, icbf, icf]
        :return:
        """
        self.target_playlist = target_playlist
        self.target_tracks = target_tracks

        # initialization
        ucf = UCF()
        ucbf = UCBF()

        icf = ICF()
        icbf = ICBF()

        # work on playlist side
        # implement ucbf with attribute: owner, title and tracks
        attr = [1, 1, 0, 0, 0]
        weight = [0.6, 0.3, 1, 1, 1]
        GROUP = [0,0,0]

        ucm = build_ucm(ds, attr, weight, GROUP)
        ucm_t = ucm.transpose().tocsr()
        ucm_t = hstack([ucm_t, urm])

        ucbf.fit(urm, ucm_t, target_playlist, target_tracks, ds, 95, 50)
        eR_ucbf = ucbf.get_estimate_rating()
        self.eR_ucbf = eR_ucbf

        # implement ucf
        ucf.fit(urm, target_playlist, target_tracks, ds, 95, 50)
        eR_ucf = ucf.get_estimate_rating()
        self.eR_ucf = eR_ucf


        # work on track side
        # implement icbf with attribute: album, artist_id
        attr = [1, 1, 0, 0, 0]
        weight = [1, 1, 1, 1, 1]

        icm = build_icm(ds, attr, weight)
        icbf.fit(urm, icm, target_playlist, target_tracks, ds, 95, 1)
        eR_icbf = icbf.get_estimate_rating()
        self.eR_icbf = eR_icbf

        # implement icf
        icf.fit(urm, target_playlist, target_tracks, ds, 95, 1)
        eR_icf = icf.get_estimate_rating()
        self.eR_icf = eR_icf

        self.e_rating = coefficient[0] * self.eR_ucbf + coefficient[1] * self.eR_ucf +\
            coefficient[2] * self.eR_icbf + coefficient[3] * self.eR_icf

    def get_estimate_rating(self):
        return self.e_rating

    def get_eR_ucf(self):
        return self.eR_ucf

    def get_eR_ucbf(self):
        return self.eR_ucbf

    def get_eR_icf(self):
        return  self.eR_icf

    def get_eR_icbf(self):
        return self.eR_icbf

    def submit_result(self):
        name = "Hybrid"
        recs = predict(self.e_rating, self.target_playlist, self.target_tracks)
        write_submission(name, self.target_playlist, recs)






