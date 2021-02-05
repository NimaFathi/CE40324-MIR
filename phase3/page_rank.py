from os import listdir

import numpy as np
import json


def page_rank(file_name, convergence_limit: float = 1):
    data = pd.read_json(file_name)
    matrix = np.zeros((len(data), len(data)))
    alpha = 0.1
    for index in range(len(data)):
        row = data.iloc[index]
        refs = set(int(item) for item in row['references'])
        ref_cells = data[data['id'].apply(lambda x: x in refs)].index
        matrix[index] = (alpha if len(ref_cells) else 1) / (len(data) - len(ref_cells))
        if len(ref_cells):
            matrix[index, ref_cells] = (1 - alpha) / len(ref_cells)
    scores = np.random.rand(matrix.shape[0], 1)
    scores = scores / np.linalg.norm(scores, 1)
    distance = np.inf
    while distance > convergence_limit:
        prev_mat = scores
        scores = matrix.T @ scores
        distance = np.linalg.norm(scores - prev_mat)
    ranks = np.argsort(-scores.reshape(-1))
    data['page-rank'] = ranks
    return data.iloc[np.argsort(ranks)]
    # creating matrix
    # nodes = {}
    # ids = list(nodes.keys())
    # size = len(ids)
    # vec = []
    # for i in nodes:
    #     interior = []
    #     neighbor = i.references
    #     for j in range(size):
    #         cap = len(neighbor)
    #         if ids[j] in neighbor:
    #             interior.append((1 - alpha) / cap)
    #         else:
    #             interior.append(alpha / (size - cap))
    #     vec.append(interior)
    # matrix = np.matrix(vec).T
    # # matrix created
    # # getting vector res
    # res = np.random.rand(matrix.shape[1], 1)
    # res = res / np.linalg.norm(res, 1)
    # distance = convergence_limit + 10
    # while distance > convergence_limit:
    #     prev_mat = res
    #     res = matrix @ res
    #     distance = np.linalg.norm(res - prev_mat)
    # vector = list(res.T)
    # sorted_vec = list(sorted(vector, reverse=True))
    # best = []
    # ids = list(nodes.keys())
    # for i in range(top_count):
    #     best.append(nodes[ids[vector.index(sorted_vec[i])]].title)
    # return best
