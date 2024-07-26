import json
import concurrent
import os
import numpy as np
from functools import partial
from enum import Enum, auto
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
import copy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from utils import *
from _evaluation import _turn_solution_scores_into_choose_count, _estimator, _estimate_pass_at_k
from evaluation import original_pass_at_k, sorted_pass_at_k_top_hat_n

class InitMethod(Enum):
    UNIFORM = auto()
    LEXICAL = auto()
    SEMANTIC = auto()  # Ranked by weighted cardinality of consensus set

class VertexType(Enum):
    code = auto()
    spec = auto()
    testcase = auto()

class VertexGroup:
    def __init__(self, task_id,  data_type, vertex_type: VertexType):
        self.data_type = data_type
        self.task_id = task_id
        self.type = vertex_type

        self.vertex_group = []

    @staticmethod
    def init_vertex_group(task_id, data_type, vertex_type: VertexType, max_num: int):
        if vertex_type == VertexType.code:
            return CodeVertexGroup(task_id, data_type, max_num)
        elif vertex_type == VertexType.spec:
            return SpecVertexGroup(task_id, data_type, max_num)
        elif vertex_type == VertexType.testcase:
            return TestcaseVertexGroup(task_id, data_type, max_num)
    
    def get_uniform_s0(self):
        return np.ones(len(self.vertex_group)) / len(self.vertex_group)
    
    def get_lexical_s0(self):
        def bleu_score(hypothesis, reference):
            smoothing = SmoothingFunction()  
            return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing.method1)

        # calculate risk matrix with bleu score
        risk = np.ones((len(self.vertex_group), len(self.vertex_group)))
        for i in range(len(self.vertex_group) - 1):
            for j in range(i + 1, len(self.vertex_group)):
                risk[i][j] = bleu_score(self.vertex_group[i], self.vertex_group[j])
                risk[j][i] = risk[i][j]

        # calculate lexical s0
        return np.array([np.sum(risk[i]) for i in range(len(self.vertex_group))]) / len(self.vertex_group)
    
    def __len__(self):
        return len(self.vertex_group)


class CodeVertexGroup(VertexGroup):
    def __init__(self, task_id, data_type, max_num):
        super().__init__(task_id, data_type, VertexType.code)
        # load code data
        data = CODE_VERTEX_GROUP_DATA
        for item in data:
            if item['task_id'] == task_id:
                break
        else:
            raise Exception(f"Cannot find code of task_id {task_id}")
        self.vertex_group = item['completions'][:max_num]


class SpecVertexGroup(VertexGroup):
    def __init__(self, task_id, data_type, max_num):
        super().__init__(task_id, data_type, VertexType.spec)
        # load spec data
        data = SPEC_VERTEX_GROUP_DATA
        for item in data:
            if item['task_id'] == task_id:
                break
        else:
            raise Exception(f"Cannot find spec of task_id {task_id}")
        self.vertex_group = item['spec_code_choices'][:max_num]


class TestcaseVertexGroup(VertexGroup):
    def __init__(self, task_id, data_type, max_num):
        super().__init__(task_id, data_type, VertexType.testcase)
        # load testcase data
        data = TESTCASE_VERTEX_GROUP_DATA
        for item in data:
            if item['task_id'] == task_id:
                break
        else:
            raise Exception(f"Cannot find testcase of task_id {task_id}")
        self.vertex_group = item['tc_input_output'][:max_num]

    
    def get_lexical_s0(self):
        # calculate risk matrix with bleu score
        risk = np.ones((len(self.vertex_group), len(self.vertex_group)))
        for i in range(len(self.vertex_group) - 1):
            for j in range(i + 1, len(self.vertex_group)):
                risk[i][j] = self.vertex_group[i] == self.vertex_group[j] 
                risk[j][i] = risk[i][j]
        # calculate lexical s0
        return np.array([np.sum(risk[i]) for i in range(len(self.vertex_group))]) / len(self.vertex_group)
                

class EdgeGroup:
    def __init__(self, task_id, data_type, vertex_group1, vertex_group2):
        self.data_type = data_type
        self.task_id = task_id
        self.vertex_group1 = vertex_group1
        self.vertex_group2 = vertex_group2
    
    @staticmethod
    def init_edge_group(task_id, data_type, vertex_group1, vertex_group2):
        if isinstance(vertex_group1, CodeVertexGroup) and isinstance(vertex_group2, SpecVertexGroup):
            return CodeSpecEdgeGroup(task_id, data_type, vertex_group1, vertex_group2)
        elif isinstance(vertex_group1, CodeVertexGroup) and isinstance(vertex_group2, TestcaseVertexGroup):
            return CodeTestcaseEdgeGroup(task_id, data_type, vertex_group1, vertex_group2)
        elif isinstance(vertex_group1, SpecVertexGroup) and isinstance(vertex_group2, TestcaseVertexGroup):
            return SpecTestcaseEdgeGroup(task_id, data_type, vertex_group1, vertex_group2)
    
    def get_adj_matrix(self):
        return self.adj_matrix


class CodeSpecEdgeGroup(EdgeGroup):
    def __init__(self, task_id,  data_type, vertex_group1, vertex_group2):
        super().__init__(task_id, data_type, vertex_group1, vertex_group2)
        # load verification results
        data = CODE_SPEC_EDGE_GROUP_DATA
        verification_results = []
        for item in data:
            if item['task_id'] == task_id:
                verification_results.append(item)
        if len(verification_results) == 0:
            raise Exception(f"Cannot find code-spec verification result of task_id {task_id}")
        
        # create nested dict for verification results
        # indexed by code str and spec str
        code_spec_res = defaultdict(dict)
        for item in verification_results:
            code_str = item['solution']
            spec_str = item['spec']
            code_spec_res[code_str][spec_str] = sum(item['result']) / len(item['result']) # average over all casual inputs

        # calculate adjacency matrix
        self.adj_matrix = np.zeros((len(vertex_group1), len(vertex_group2)))
        for i in range(len(vertex_group1)):
            for j in range(len(vertex_group2)):
                code_str = vertex_group1.vertex_group[i]
                spec_str = vertex_group2.vertex_group[j]
                if code_str in code_spec_res and spec_str in code_spec_res[code_str]:
                    self.adj_matrix[i][j] = code_spec_res[code_str][spec_str]
                else:
                    self.adj_matrix[i][j] = 0
                    print("Some code-spec pair is not found in verification results")

class CodeTestcaseEdgeGroup(EdgeGroup):
    def __init__(self, task_id, data_type, vertex_group1, vertex_group2):
        super().__init__(task_id, data_type, vertex_group1, vertex_group2)
        # load verification results
        data = CODE_TESTCASE_EDGE_GROUP_DATA
        verification_results = []
        for item in data:
            if item['task_id'] == task_id:
                verification_results.append(item)
        if len(verification_results) == 0:
            raise Exception(f"Cannot find code-testcase verification result of task_id {task_id}")
        
        # create nested dict for verification results
        # indexed by code str and testcase idx
        code_testcase_res = defaultdict(dict)
        for item in verification_results:
            code_str = item['code']
            for i, res in enumerate(item['result']):
                code_testcase_res[code_str][i] = float(res)

        # calculate adjacency matrix
        self.adj_matrix = np.zeros((len(vertex_group1), len(vertex_group2)))
        for i in range(len(vertex_group1)):
            for j in range(len(vertex_group2)):
                code_str = vertex_group1.vertex_group[i]
                testcase_idx = j
                if code_str in code_testcase_res and testcase_idx in code_testcase_res[code_str]:
                    self.adj_matrix[i][j] = code_testcase_res[code_str][testcase_idx]
                else:
                    self.adj_matrix[i][j] = 0
                    print("Some code-testcase pair is not found in verification results")


class SpecTestcaseEdgeGroup(EdgeGroup):
    def __init__(self, task_id, data_type, vertex_group1, vertex_group2):
        super().__init__(task_id, data_type, vertex_group1, vertex_group2)
        # load verification results
        data = SPEC_TESTCASE_EDGE_GROUP_DATA
        verification_results = []
        for item in data:
            if item['task_id'] == task_id:
                verification_results.append(item)
        if len(verification_results) == 0:
            raise Exception(f"Cannot find spec-testcase verification result of task_id {task_id}")
        
        # create nested dict for verification results
        # indexed by spec str and testcase idx
        spec_testcase_res = defaultdict(dict)
        for item in verification_results:
            spec_str = item['spec']
            for i, res in enumerate(item['result']):
                spec_testcase_res[spec_str][i] = float(res)
        
        # calculate adjacency matrix
        self.adj_matrix = np.zeros((len(vertex_group1), len(vertex_group2)))
        for i in range(len(vertex_group1)):
            for j in range(len(vertex_group2)):
                spec_str = vertex_group1.vertex_group[i]
                testcase_idx = j
                if spec_str in spec_testcase_res and testcase_idx in spec_testcase_res[spec_str]:
                    self.adj_matrix[i][j] = spec_testcase_res[spec_str][testcase_idx]
                else:
                    self.adj_matrix[i][j] = 0
                    print("Some spec-testcase pair is not found in verification results")


class NPartiteGraph:
    def __init__(self, task_id: int, data_type, vertex_type_lst, max_iter=5000, init_method=InitMethod.LEXICAL):
        self.data_type = data_type
        self.task_id = task_id
        self.vertex_type_lst = vertex_type_lst
        self.max_iter = max_iter

        # load vertex data
        self.vertex_group_lst = []
        self.vertex_type2idx_range = {} # Each vertex type is assigned a range of indices in the `self.score_vector`
        for vt in VertexType:
            for used_vt, max_num in vertex_type_lst:
                if vt == used_vt:
                    break
            else:
                continue
            self.vertex_group_lst.append(VertexGroup.init_vertex_group(task_id, data_type, vt, max_num))
            self.vertex_type2idx_range[vt] = (sum([len(vg) for vg in self.vertex_group_lst[:-1]]), sum([len(vg) for vg in self.vertex_group_lst]))

        # calculate adjacency matrix
        self.edge_group_lst = defaultdict(dict)
        for i in range(len(self.vertex_group_lst)):
            for j in range(i+1, len(self.vertex_group_lst)):
                eg = EdgeGroup.init_edge_group(task_id, data_type, self.vertex_group_lst[i], self.vertex_group_lst[j])
                self.edge_group_lst[i][j] = eg
        # concatenate all edges into a adjancency matrix
        num_vertex = sum([len(vg) for vg in self.vertex_group_lst])
        self.adj_matrix = np.empty((0, num_vertex), dtype=np.float32)
        for i in range(len(self.vertex_group_lst)):
            matrix_lst = []
            if i > 0:
                tmp = np.concatenate([self.edge_group_lst[j][i].get_adj_matrix().T for j in range(i)], axis=1)
                matrix_lst.append(tmp)
            tmp = np.zeros((len(self.vertex_group_lst[i]), len(self.vertex_group_lst[i])))
            matrix_lst.append(tmp)
            if i < len(self.vertex_group_lst) - 1:
                tmp = np.concatenate([eg.get_adj_matrix() for eg in self.edge_group_lst[i].values()], axis=1)
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
        self.initialize_s0(init_method)
        # mask then normalize s0
        self.s0 = np.where(self.mask_vector, 0, self.s0)
        for b,e in self.vertex_type2idx_range.values():
            total = np.sum(self.s0[b:e])
            if total != 0:
                self.s0[b:e] = self.s0[b:e] / total

        # initialize score
        self.score_vector = copy.deepcopy(self.s0)

        # dynamic alpha (see Appendix)
        num_edges = 0
        for i in range(len(self.vertex_group_lst)):
            for j in range(i+1, len(self.vertex_group_lst)):
                num_edges += len(self.vertex_group_lst[i]) * len(self.vertex_group_lst[j])
        mean_edge_weight = np.sum(self.adj_matrix) / 2 / num_edges
        self.alpha = 0.01 if mean_edge_weight < 0.16 else 0.95

    def fit(self):
        iter_left = self.max_iter
        prev_score = None
        while (prev_score is None or np.linalg.norm(self.score_vector - prev_score) > 1e-6) and iter_left > 0:
            prev_score = self.score_vector
            self.score_vector = self.alpha * np.matmul(self.transition_matrix, self.score_vector) + (1-self.alpha) * self.s0
            iter_left -= 1
        if iter_left == 0:
            print(f"Warning: Does not converge in {self.max_iter} iterations")
        return self.max_iter-iter_left

    def get_sorted_partition_score(self, vertex_type: VertexType):
        """
        Return a list of (vertex_string, score) tuples, sorted by score in descending order
        """
        for vg in self.vertex_group_lst:
            if vg.type == vertex_type:
                break
        vertex_string_lst = vg.vertex_group

        # Assign vertices without any edges to 0 score
        score_vector = self.score_vector.copy()
        score_vector[self.mask_vector == 1] = 0
        score_lst = score_vector[self.vertex_type2idx_range[vertex_type][0]:self.vertex_type2idx_range[vertex_type][1]].tolist()

        assert len(vertex_string_lst) == len(score_lst)
        return sorted(zip(vertex_string_lst, score_lst), key=lambda x: x[1], reverse=True)
    
    def initialize_s0(self, init_method=InitMethod.UNIFORM):
        if init_method == InitMethod.UNIFORM:
            self.s0 = np.concatenate([vg.get_uniform_s0() for vg in self.vertex_group_lst], axis=0)
        elif init_method == InitMethod.LEXICAL:
            self.s0 = np.concatenate([vg.get_lexical_s0() for vg in self.vertex_group_lst], axis=0)
        elif init_method == InitMethod.SEMANTIC:
            adj_matrix = self.adj_matrix.copy()

            s0_lst = []
            for b,e in self.vertex_type2idx_range.values():
                unique_rows, counts = np.unique(adj_matrix[b:e], axis=0, return_counts=True)
                # score = np.log(counts + 1)
                score = counts
                for tmpb, tmpe in self.vertex_type2idx_range.values():
                    # skip the diagonal block or the missing block 
                    if tmpb == b or tmpb == tmpe:
                        continue
                    sum_weight = unique_rows[:,tmpb:tmpe].sum(axis=1) + 1 # add 1 to avoid 0
                    normalized_weight = sum_weight / (tmpe - tmpb) # normalize to keep weight from different vertex types comparable
                    score = score * normalized_weight # score of vertex i is calculated as: \Product_t #Agreement of i with type t vertices
            
                idx_lst = []
                for row in adj_matrix[b:e]:
                    idx = np.where((unique_rows == row).all(axis=1))
                    assert len(idx) == 1
                    idx_lst.append(idx[0][0])
                    assert np.array_equal(unique_rows[idx[0][0]], row)
                
                s0 = score[idx_lst]
                total = np.sum(s0)
                if total != 0:
                    s0 = s0 / total
                s0_lst.append(s0)
            
            self.s0 = np.concatenate(s0_lst, axis=0)
        

def process_fn(task_id, data_type, vertex_type_lst, init_method):
    graph = NPartiteGraph(task_id=task_id, data_type=data_type, vertex_type_lst=vertex_type_lst, init_method=init_method)
    num_iteration = graph.fit() 
    code_score = graph.get_sorted_partition_score(VertexType.code)

    return task_id, code_score, num_iteration


def get_code_sorted(data_type, vertex_type_lst, init_method):
    # print("Start processing {} ......".format(data_type))
    dataset = load_jsonl(f"../data/{data_type}.jsonl")
    task_ids_lst = [item['task_id'] for item in dataset]

    code_score_dict = {}

    fn = partial(process_fn, data_type=data_type, vertex_type_lst=vertex_type_lst, init_method=init_method)

    code_score_dict = {}


    # for task_id in tqdm(task_ids_lst):
    #     rtn = fn(task_id)
    #     code_score_dict[rtn[0]] = rtn[1]

    import time
    start_time = time.time()

    num_iteration = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(fn, task_id) for task_id in task_ids_lst]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        res = future.result()
        code_score_dict[res[0]] = res[1]
        num_iteration.append(res[2])
    
    end_time = time.time()
    print(f"Avg Time: {(end_time - start_time) / len(task_ids_lst)} seconds")

    # save sorted code
    dump_pickle(code_score_dict, f"../save/{data_type}/code_sorted_npartite.pkl")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    for data_type, isplus in [("human_eval", False), ("human_eval", True), ("CodeContests", False), ("mbpp", False)]:
        print(f"==== {data_type}{'+' if isplus else ''} ====")
        # baseline pass@k without sorting
        pass_origin = [str(x) for x in original_pass_at_k(data_type, isplus).values()]
        print("Baseline: pass@1/2/5: ", '/'.join(pass_origin))

        vertex_type_lst = [(VertexType.code, 200), (VertexType.spec, 50), (VertexType.testcase, 100)]

        CODE_VERTEX_GROUP_DATA = load_jsonl(f"../save/{data_type}/solutions.jsonl")
        SPEC_VERTEX_GROUP_DATA = load_jsonl(f"../save/{data_type}/specs.jsonl")
        TESTCASE_VERTEX_GROUP_DATA = load_pickle(f"../save/{data_type}/test_cases.pkl")
        CODE_SPEC_EDGE_GROUP_DATA = load_pickle(f"../save/{data_type}/code_spec_res.pkl")
        CODE_TESTCASE_EDGE_GROUP_DATA = load_pickle(f"../save/{data_type}/code_testcase_res.pkl")
        SPEC_TESTCASE_EDGE_GROUP_DATA = load_pickle(f"../save/{data_type}/spec_testcase_res.pkl")

        init_method = InitMethod.LEXICAL
        get_code_sorted(data_type, vertex_type_lst, init_method)
        pass_tophatn = [str(x) for x in sorted_pass_at_k_top_hat_n(data_type, isplus).values()]
        print("MPSC-Lexical: pass@1/2/5: ", '/'.join(pass_tophatn))

        init_method = InitMethod.SEMANTIC
        get_code_sorted(data_type, vertex_type_lst, init_method)
        pass_tophatn = [str(x) for x in sorted_pass_at_k_top_hat_n(data_type, isplus).values()]
        print("MPSC-Semantic: pass@1/2/5: ", '/'.join(pass_tophatn))