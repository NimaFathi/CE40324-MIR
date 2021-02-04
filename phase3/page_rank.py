from os import listdir

import numpy as np
import json


def page_rank(alpha: float = 0.1, convergence_limit: float = 1, top_count: int = 10):
    # creating matrix
    nodes = {}
    ids = list(nodes.keys())
    size = len(ids)
    vec = []
    for i in nodes:
        interior = []
        neighbor = i.references
        for j in range(size):
            cap = len(neighbor)
            if ids[j] in neighbor:
                interior.append((1 - alpha) / cap)
            else:
                interior.append(alpha / (size - cap))
        vec.append(interior)
    matrix = np.matrix(vec).T
    # matrix created
    # getting vector res
    res = np.random.rand(matrix.shape[1], 1)
    res = res / np.linalg.norm(res, 1)
    distance = convergence_limit + 10
    while distance > convergence_limit:
        prev_mat = res
        res = matrix @ res
        distance = np.linalg.norm(res - prev_mat)
    vector = list(res.T)
    sorted_vec = list(sorted(vector, reverse=True))
    best = []
    ids = list(nodes.keys())
    for i in range(top_count):
        best.append(nodes[ids[vector.index(sorted_vec[i])]].title)
    return best
