import copy
import json
import os
from collections import defaultdict
from enum import Enum, auto
from typing import List

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class IntraConsistency(Enum):
    UNIFORM = auto()
    LEXICAL = auto()
    SEMANTIC = auto()  # Ranked by weighted cardinality of consensus set

# The user defined perspectives
class VertexType(Enum):
    code = auto()
    spec = auto()
    testcase = auto()

class VertexGroup:
    def __init__(self, vertex_type: VertexType, vertex_path: str, max_num: int):
        self.type = vertex_type
        data = json.load(open(vertex_path, 'r'))
        self.vertex_group = data[:max_num]

    def get_uniform_s0(self):
        return np.ones(len(self.vertex_group)) / len(self.vertex_group)
    
    def get_mbr_s0(self):
        def bleu_score(hypothesis, reference):
            smoothing = SmoothingFunction()  
            return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing.method1)

        # calculate risk matrix with bleu score
        risk = np.ones((len(self.vertex_group), len(self.vertex_group)))
        for i in range(len(self.vertex_group) - 1):
            for j in range(i + 1, len(self.vertex_group)):
                risk[i][j] = bleu_score(self.vertex_group[i], self.vertex_group[j])
                risk[j][i] = risk[i][j]
        # calculate mbr
        return np.array([np.sum(risk[i]) for i in range(len(self.vertex_group))]) / len(self.vertex_group)
    
    def __len__(self):
        return len(self.vertex_group)

class EdgeGroup:
    def __init__(self, vertex_group1: VertexGroup, vertex_group2: VertexGroup, verification_path=None):
        """
        Args:
            vertex_group1 (VertexGroup): the first vertex group
            vertex_group2 (VertexGroup): the second vertex group
            verification_path (str): path to verification results.
        """
        self.vertex_group1 = vertex_group1
        self.vertex_group2 = vertex_group2

        # load verification results
        inter_consistency_matrix = json.load(open(verification_path, 'r'))
        
        # calculate adjacency matrix
        self.adj_matrix = np.zeros((len(vertex_group1), len(vertex_group2)))
        for i in range(len(vertex_group1)):
            for j in range(len(vertex_group2)):
                v1 = vertex_group1.vertex_group[i]
                v2 = vertex_group2.vertex_group[j]
                if v1 in inter_consistency_matrix and v2 in inter_consistency_matrix[v1]:
                    self.adj_matrix[i][j] = inter_consistency_matrix[v1][v2]
                else:
                    self.adj_matrix[i][j] = 0
                    print(f"=======Inter-consistency of one edge between {self.vertex_group1.type.name} and {self.vertex_group2.type.name} is not found in {verification_path}=======")
                    print(f"{self.vertex_group1.type.name} vertex: {v1}")
                    print(f"{self.vertex_group2.type.name} vertex: {v2}")
        
    def get_adj_matrix(self):
        return self.adj_matrix
    
class NPartiteGraph:
    def __init__(self, vertex_group_lst, edge_group_lst, max_iter=5000, intra_consistency_measure=IntraConsistency.LEXICAL, alpha_minor=0.01, alpha_major=0.95, threshold=0.16):
        self.max_iter = max_iter

        # load vertex data
        self.vertex_group_lst = vertex_group_lst
        self.vertex_type2idx_range = {} # Each vertex type is assigned a range of indices in the `self.score_vector`
        self.vertex_type2id = {} # Each vertex type is assigned an id
        cur_idx = 0
        for i, vg in enumerate(self.vertex_group_lst):
            cur_length = len(vg)
            self.vertex_type2idx_range[vg.type] = (cur_idx, cur_idx + cur_length)
            self.vertex_type2id[vg.type] = i
            cur_idx += cur_length

        # load edge data
        self.edge_group_dict = defaultdict(dict)
        assert len(edge_group_lst) == len(self.vertex_group_lst) * (len(self.vertex_group_lst) - 1) / 2, "Number of edge groups should be n(n-1)/2, where n is the number of vertex groups"
        for eg in edge_group_lst:
            self.edge_group_dict[self.vertex_type2id[eg.vertex_group1.type]][self.vertex_type2id[eg.vertex_group2.type]] = eg

        # concatenate all edges into a adjancency matrix
        num_vertex = sum([len(vg) for vg in self.vertex_group_lst])
        self.adj_matrix = np.empty((0, num_vertex), dtype=np.float32)
        for i in range(len(self.vertex_group_lst)):
            matrix_lst = []
            if i > 0:
                tmp = np.concatenate([self.edge_group_dict[j][i].get_adj_matrix().T for j in range(i)], axis=1)
                matrix_lst.append(tmp)
            tmp = np.zeros((len(self.vertex_group_lst[i]), len(self.vertex_group_lst[i])))
            matrix_lst.append(tmp)
            if i < len(self.vertex_group_lst) - 1:
                tmp = np.concatenate([eg.get_adj_matrix() for eg in self.edge_group_dict[i].values()], axis=1)
                matrix_lst.append(tmp)
            self.adj_matrix = np.concatenate([self.adj_matrix, np.concatenate(matrix_lst, axis=1)], axis=0)
        
        # calculate transition matrix
        self.degree_vector = np.sum(self.adj_matrix, axis=1)
        self.mask_vector = np.where(self.degree_vector == 0, 1, 0)
        masked_degree_vector = np.where(self.degree_vector == 0, 1, self.degree_vector)  #avoid division by zero
        masked_norm_vector = np.power(masked_degree_vector, -1/2)
        norm_vector = np.where(self.mask_vector, 0, masked_norm_vector)
        norm_matrix = np.diag(norm_vector)
        self.transition_matrix = np.matmul(norm_matrix, np.matmul(self.adj_matrix, norm_matrix))
        
        # initialize s0 vector
        self.s0 = None
        self.initialize_s0(intra_consistency_measure) # initialize s0 with intra-consistency measures

        # initialize score
        self.score_vector = copy.deepcopy(self.s0)

        # set alpha
        num_edges = 0
        for i in range(len(self.vertex_group_lst)):
            for j in range(i+1, len(self.vertex_group_lst)):
                num_edges += len(self.vertex_group_lst[i]) * len(self.vertex_group_lst[j])
        mean_edge_weight = np.sum(self.adj_matrix) / 2 / num_edges
        self.alpha = alpha_minor if mean_edge_weight < threshold else alpha_major

    def fit(self):
        max_iter = self.max_iter
        prev_score = None
        while (prev_score is None or np.linalg.norm(self.score_vector - prev_score) > 1e-6) and max_iter > 0:
            prev_score = self.score_vector
            self.score_vector = self.alpha * np.matmul(self.transition_matrix, self.score_vector) + (1-self.alpha) * self.s0
            max_iter -= 1
        if max_iter == 0:
            print(f"Warning: Does not converge in {self.max_iter} iterations")
        return 5000-max_iter

    def get_vertex_group_score(self, vertex_type: VertexType):
        """
        Return a list of (vertex_string, score) tuples, sorted by score in descending order
        """
        vg = self.vertex_group_lst[self.vertex_type2id[vertex_type]]
        vertex_string_lst = vg.vertex_group

        # Assign vertices without any edges to 0 score
        score_vector = self.score_vector.copy()
        score_vector[self.mask_vector == 1] = 0
        score_lst = score_vector[self.vertex_type2idx_range[vertex_type][0]:self.vertex_type2idx_range[vertex_type][1]].tolist()

        assert len(vertex_string_lst) == len(score_lst)
        return sorted(zip(vertex_string_lst, score_lst), key=lambda x: x[1], reverse=True)
    
    def initialize_s0(self, init_method=IntraConsistency.UNIFORM):
        if init_method == IntraConsistency.UNIFORM:
            self.s0 = np.concatenate([vg.get_uniform_s0() for vg in self.vertex_group_lst], axis=0)
        elif init_method == IntraConsistency.LEXICAL:
            self.s0 = np.concatenate([vg.get_mbr_s0() for vg in self.vertex_group_lst], axis=0)
        elif init_method == IntraConsistency.SEMANTIC:
            adj_matrix = self.adj_matrix.copy()

            s0_lst = []
            # for each vertex group
            for b,e in self.vertex_type2idx_range.values():
                # find structural equivalence class, each corresponds to a unique row
                unique_rows, counts = np.unique(adj_matrix[b:e], axis=0, return_counts=True)
                score = counts

                # calculate the score of each structural equivalence class
                for tmpb, tmpe in self.vertex_type2idx_range.values():
                    # skip the diagonal block 
                    if tmpb == b:
                        continue
                    sum_weight = unique_rows[:,tmpb:tmpe].sum(axis=1)
                    normalized_weight = sum_weight / (tmpe - tmpb) # normalize to keep weight from different vertex types comparable
                    score = score * normalized_weight # score of vertex i is calculated as: \Product_t #Agreement of i with type t vertices
            
                # map each vertex to its structural equivalence class
                idx_lst = []
                for row in adj_matrix[b:e]:
                    idx = np.where((unique_rows == row).all(axis=1))
                    assert len(idx) == 1
                    idx_lst.append(idx[0][0])
                    assert np.array_equal(unique_rows[idx[0][0]], row)
                
                # normalization over each vertex group
                s0 = score[idx_lst]
                total = np.sum(s0)
                if total != 0:
                    s0 = s0 / total
                s0_lst.append(s0)
            
            self.s0 = np.concatenate(s0_lst, axis=0)
        
def run(vertex_group_lst: List[VertexGroup], edge_group_lst: List[EdgeGroup], target_vertex_type: VertexType, intra_consistency_measure: IntraConsistency, alpha: float):
    """
    Args:
        vertex_group_lst (list(VertexGroup)): a list of vertex groups
        edge_group_lst (list(EdgeGroup)): a list of edge groups
        target_vertex_type (VertexType): the vertex type to be ranked
        intra_consistency_measure (IntraConsistency): the measure to calculate intra-consistency
        alpha (float): the factor control the weight of inter-consistency and intra-consistency
    Returns:
        score_lst (list(tuple(str, float))): the target vertices sorted by score in descending order. each item contains a string corresponding to one output and its score
        num_iteration (int): number of iterations for convergence
    """
    graph = NPartiteGraph(vertex_group_lst, edge_group_lst, intra_consistency_measure=intra_consistency_measure, alpha=alpha)
    num_iteration = graph.fit() 
    score_lst = graph.get_vertex_group_score(target_vertex_type)

    return score_lst, num_iteration


if __name__ == '__main__':
    # Initialize vertex groups
    vertex_group_lst = []
    vertex_group_lst.append(VertexGroup(VertexType.code, 'data/vertex/code.json', 200))
    vertex_group_lst.append(VertexGroup(VertexType.spec, 'data/vertex/spec.json', 50))
    vertex_group_lst.append(VertexGroup(VertexType.testcase, 'data/vertex/testcase.json', 100))

    # Initialize edge groups
    edge_group_lst = []
    edge_group_lst.append(EdgeGroup(vertex_group_lst[0], vertex_group_lst[1], 'data/edge/code-spec.json'))
    edge_group_lst.append(EdgeGroup(vertex_group_lst[0], vertex_group_lst[2], 'data/edge/code-testcase.json'))
    edge_group_lst.append(EdgeGroup(vertex_group_lst[1], vertex_group_lst[2], 'data/edge/spec-testcase.json'))

    # Run graph algorithm
    score_lst, num_iteration = run(vertex_group_lst, edge_group_lst, target_vertex_type=VertexType.code, intra_consistency_measure=IntraConsistency.SEMANTIC)