from scipy.sparse import *
from scipy.sparse.linalg import norm as snorm
import numpy as np


def cosine_item(matrix, target_list, shrinkage, knn):
    """
    Compute track similarity
    :param matrix: row: track attributes
                   column: track index
    :param target_list: target tracks
    :param shrinkage: shrinkage factor
    :param knn: k-Nearest neighbour
    :return: similarity matrix S
    """
    matrix = matrix.tocsr()
    M = matrix.copy()

    # compute norm for each column
    norm = snorm(M, axis=0)
    norm[(norm == 0)] = 1

    # apply normalization
    M = M.multiply(csr_matrix(1 / norm))
    M_t = M.transpose()
    M_t = M_t.tocsr()

    # clean the matrix with target list
    M_t = M_t[[x for x in target_list]]

    # compute cosine similarity
    S_cos = M_t.dot(M)

    # apply shrinkage by multiplying F = #commen_features/(#commen_features + 1)
    M_norm = M.copy()
    M_t_norm = M_t.copy()
    M_norm[M_norm.nonzero()] = 1
    M_t_norm[M_t_norm.nonzero()] = 1
    F_num = M_t_norm.dot(M_norm)
    F_den = F_num.copy()
    F_den.data = 1 / (F_den.data + shrinkage)

    S = S_cos.multiply(F_num).multiply(F_den)

    # apply knn
    for i in range(0, S.shape[0]):
        row = S.data[S.indptr[i]:S.indptr[i + 1]]
        sorted_idx = row.argsort()[:-knn]
        row[sorted_idx] = 0

    # set zero to diagonal
    S.setdiag(0)
    S.eliminate_zeros()

    # normalize S
    s_norm = S.sum(axis=1)
    s_norm[(s_norm == 0)] = 1
    S = S.multiply(csr_matrix(1 / s_norm))

    return S


def cosine_user(matrix, target_list, shrinkage, knn):
    """
    Compute playlist similarity
    :param matrix: row: playlist
                   column: playlist attributes
    :param target_list: target playlist
    :param shrinkage: shrinkage factor
    :param knn: k nearest neighbour
    :return: S row: target playlist
               column: total playlist
    """
    matrix = matrix.tocsr()
    M = matrix.copy()

    # compute norm for each column
    norm = snorm(M, axis=1)
    norm[(norm == 0)] = 1

    # apply normalization
    M = M.multiply((csr_matrix(1 / norm).transpose()).tocsr())
    M_t = M.transpose()
    M_t = M_t.tocsr()

    # clean the matrix with target list
    M = M[[x for x in target_list]]

    # compute cosine similarity
    S_cos = M.dot(M_t)

    # apply shrinkage by multiplying F = #commen_features/(#commen_features + 1)
    M_norm = M.copy()
    M_t_norm = M_t.copy()
    M_norm[M_norm.nonzero()] = 1
    M_t_norm[M_t_norm.nonzero()] = 1
    F_num = M_norm.dot(M_t_norm)
    F_den = F_num.copy()
    F_den.data = 1 / (F_den.data + shrinkage)

    S = S_cos.multiply(F_num).multiply(F_den)

    # apply knn
    for i in range(0, S.shape[0]):
        row = S.data[S.indptr[i]:S.indptr[i + 1]]
        sorted_idx = row.argsort()[:-knn]
        row[sorted_idx] = 0

    # set zero to diagonal
    S.setdiag(0)
    S.eliminate_zeros()

    # normalize S
    s_norm = S.sum(axis=1)
    s_norm[(s_norm == 0)] = 1
    S = S.multiply(csr_matrix(1 / s_norm))

    return S