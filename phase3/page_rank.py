from os import listdir
import pandas as pd
import numpy as np
import json


def page_rank(file_name, convergence_limit: float = 1):
    data = pd.read_json(file_name)
    print(data)
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


if __name__ == '__main__':
    x = page_rank('./crawled_papers.json')
    print(x)