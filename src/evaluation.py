import os
from math import comb
from execution import evaluate_with_test_code
from utils import load_jsonl, load_pickle, dump_jsonl, dump_pickle
from _evaluation import _estimate_pass_at_k

def merge_completion_and_prompt(completions, prompts, isplus=False):
    task2completions = {}
    for completion in completions:
        task2completions[completion["task_id"]] = completion["completions"]
    res = []
    for prompt in prompts:
        task_id = prompt["task_id"]
        for solution in task2completions[task_id]:
            res.append(
                {
                    "task_id": task_id,
                    "prompt": prompt["prompt"],
                    "completion": solution,
                    "test": prompt["test"] if not isplus else prompt["plus_test"],
                    "entry_point": prompt["entry_point"],
                }
            )
    return res

def extend_sorted(code_sorted, completions, rd=True):
    """
    extend the `data` to include all solutions, so that data length is 200
    Args:
        code_sorted: the sorted solutions with score. The deduplicated version, length is less equal than 200
        completions: The non-deduplicated version, length is always 200
    """
    scores = {}
    for data in completions:
        if data["task_id"] not in code_sorted:
            scores[data["task_id"]] = [(x, 0) for x in data["completions"]]
            continue
        scores[data["task_id"]] = []
        for x in data["completions"]:
            for dt in code_sorted[data["task_id"]]:
                if dt[0] == x:
                    scores[data["task_id"]].append((x, dt[1] if not rd else round(dt[1], 4)))
                    break
            else:
                scores[data["task_id"]].append((x, 0))
        scores[data["task_id"]] = sorted(scores[data["task_id"]], key=lambda x: x[1], reverse=True)
    return scores

def extend_ispassed(ispassed, completions):
    for data in completions:
        for x in data["completions"]:
            if x not in ispassed[data["task_id"]]:
                ispassed[data["task_id"]][x] = 0
    return ispassed

def original_pass_at_k(data_type, isplus=False):
    se_path = "solution_eval_plus.pkl" if isplus else "solution_eval.pkl"
    if os.path.exists(f"../save/{data_type}/{se_path}"):
        data = load_pickle(f"../save/{data_type}/{se_path}")
    else:
        completions = load_jsonl(f"../save/{data_type}/solutions.jsonl")
        prompts = load_jsonl(f"../data/{data_type}.jsonl" if not isplus else f"../data/{data_type}_plus.jsonl")

        data = evaluate_with_test_code(merge_completion_and_prompt(completions, prompts, isplus), 1)
        dump_pickle(data, f"../save/{data_type}/{se_path}")

    completions = load_jsonl(f"../save/{data_type}/solutions.jsonl")
    ispassed = {}
    for dt in data:
        if dt["task_id"] not in ispassed:
            ispassed[dt["task_id"]] = {}
        ispassed[dt["task_id"]][dt["completion"]] = dt["passed"]

    corrects = []
    totals = []
    for dt in completions:
        totals.append(len(dt["completions"]))
        corrects.append(sum([ispassed[dt["task_id"]][x] if x in ispassed[dt["task_id"]] else 0 for x in dt["completions"]]))
    
    pass_at_k = {}
    for k in [1,2,5]:
        pass_at_k[f"pass@{k}"] = round(_estimate_pass_at_k(totals, corrects, k).mean() * 100, 2)
    # print(pass_at_k)
    return pass_at_k


def sorted_pass_at_k_top_hat_n(data_type, isplus=False):
    se_path = "solution_eval_plus.pkl" if isplus else "solution_eval.pkl"
    if os.path.exists(f"../save/{data_type}/{se_path}"):
        data = load_pickle(f"../save/{data_type}/{se_path}")
    else:
        print("solution_eval.pkl not found")
        exit()
    ispassed = {}
    for dt in data:
        if dt["task_id"] not in ispassed:
            ispassed[dt["task_id"]] = {}
        ispassed[dt["task_id"]][dt["completion"]] = dt["passed"]
    task_ids = list(ispassed.keys())
    
    data = load_pickle(f"../save/{data_type}/code_sorted_npartite.pkl") # The non-deduplicated version, length is always 200
    completions = load_jsonl(f"../save/{data_type}/solutions.jsonl") # The non-deduplicated version, length is always 200

    pass_at_k = {}
    for k in [1,2,5]:
        totals = []
        corrects = []
        for task_id in task_ids:
            hat_n = k if k <= len(data[task_id]) else len(data[task_id])
            min_score = data[task_id][hat_n-1][1]
            while hat_n < len(data[task_id]):
                if data[task_id][hat_n][1] != min_score:
                    break
                hat_n += 1
            assert data[task_id][hat_n-1][1] == min_score
            totals.append(hat_n)
            corrects.append(sum([ispassed[task_id][x[0]] == 1 for x in data[task_id][:hat_n]]))
        pass_at_k[f"pass@{k}"] = round(_estimate_pass_at_k(totals, corrects, k).mean() * 100 , 2)
    return pass_at_k


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    """pass@k"""
    for data_type, isplus in [("human_eval", False), ("human_eval", True), ("CodeContests", False), ("mbpp", False)]:
        pass_origin = [str(x) for x in original_pass_at_k(data_type, isplus).values()]
        pass_tophatn = [str(x) for x in sorted_pass_at_k_top_hat_n(data_type, isplus).values()]

        print(f"{data_type}{'+' if isplus else ''} {' & '.join(pass_origin)} \\\\")
        print(f"{data_type}{'+' if isplus else ''} {' & '.join(pass_tophatn)} \\\\")