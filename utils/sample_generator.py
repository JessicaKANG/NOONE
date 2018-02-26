from utils import pickle_operations as pickle_ops
from utils import lil_operations as lil_ops


def generate_sample(surname, name):
    """
    Given the sample index, return a sample pack
    :param surname, name: index of a sample
    :return: urm, target_playlist, target_tracks, answer
    """
    path_ = "./test_data/"
    index = surname + str(name)
    _path = ".txt"

    path_ev = path_ + "ev" + index + _path
    path_urm = path_ + "urm" + index + _path + ".npz"
    path_tg = path_ + "tg" + index + _path

    answer = pickle_ops.load(path_ev)
    urm = lil_ops.load_lil(path_urm)
    target_tracks = pickle_ops.load(path_tg)
    target_playlist = list(answer.keys())

    return urm, target_playlist, target_tracks, answer

