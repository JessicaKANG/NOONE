import numpy as np
from scipy.sparse import *


def save_lil(path, m_lil):
    """
    Convert a lil_matrix into csr_matrix and save to path
    :param path: saved file path
    :param m_lil: matrix to be saved
    :return:
    """
    m_csr = m_lil.tocsr()
    np.savez(path, data=m_csr.data, indices=m_csr.indices, indptr=m_csr.indptr, shape=m_csr.shape)


def load_lil(path):
    """
    Loade a csr_matrix from path and convert to lil_matrix
    :param path: saved file path
    :return: a lil_matrix
    """
    loader = np.load(path)
    m_csr = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    m_lil = m_csr.tolil()
    return m_lil