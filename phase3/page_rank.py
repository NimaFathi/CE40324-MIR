from os import listdir
import numpy as np

class PageRank:
    def __init__(self, alpha, max_iter):
        self.alpha = alpha
        self.max_iter = max_iter
        self.paper_id_lst = None
        self.A = None
        self.init_paper_id_lst()
        self.init_adjacency_matrix()

    def init_paper_id_lst(self):
        file_name_lst = listdir('crawler/papers')
        self.paper_id_lst = [file_name[:-5] for file_name in file_name_lst]

    def init_adjacency_matrix(self):
        n = len(self.paper_id_lst)
        idx = dict(zip(self.paper_id_lst, list(range(n))))
        self.A = np.zeros((n, n))
        for paper_id in self.paper_id_lst:
            with open(f'crawler/papers/{paper_id}') as file:
                paper = json.load(file)
            for ref in paper['references']:
                if ref in idx:
                    self.A[idx[paper_id], idx[ref]] = 1.
