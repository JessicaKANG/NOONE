from scipy.sparse import *
from scipy.spatial import distance_matrix
import numpy as np
import copy


def distance(lol, target_list, shrinkage, knn):
    """
    Compute distance similarity
    :param lol: list of list
    :param target_list:
    :param shrinkage:
    :param knn:
    :return: D_cleaned
    """
    L = copy.deepcopy(lol)
    print(len(target_list))
    D = distance_matrix(L[0:4421], L)

    D = csr_matrix(D)
    #D_cleaned = D[[x for x in target_list]]

    return D
