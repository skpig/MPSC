import json
import os
import re
from tqdm import tqdm
import time
import ast
import pickle
from utils import load_jsonl, load_pickle, dump_jsonl, dump_pickle
import shutil

from api import infer_llm, infer_llm_completion
from execution import evaluate_spec_with_test_cases, evaluate_with_specs_and_casual_input, evaluate_with_test_cases
from _execution import time_limit

llm_engine = "gpt-35-turbo"

class Solution_Generator:
    """
    Generate executable code
    """
    @staticmethod
    def prompt(id, dataset):
        """
        Prompt to generate exec code with no exemplars
        """
        instruction = "I want you to act like a Python programmer. I will give you the declaration of a function and comments about its property. You need to implement the body of the function in the code block. Do not modify any code I provide. Do not provide any explanations. \n"
    
        question = "Here is the question.\n" + "```python\n" + dataset[id]['prompt'] + "```\n"
    
        return instruction, question

    @staticmethod
    def inference_all(resume_id=None):
        """
        Generate exec codes for all samples in dataset,
        """
        # load human_eval_dataset
        with open(f"../data/{data_type}.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f.readlines()]
        
        file_mode = 'a' if resume_id is not None else 'w'
        resume_id = resume_id if resume_id is not None else 0
        with open(f"../save/{data_type}/solutions.jsonl", file_mode) as f:
            # for each sample in human_eval_dataset
            for id in tqdm(range(resume_id, len(dataset))):
                prompt = dataset[id]['prompt']
                task_id = dataset[id]['task_id']
                entry_point = dataset[id]['entry_point']

                # generate prompt
                instruction, question = Solution_Generator.prompt(id, dataset)
                
                # inference with chatgpt api
                completions = []
                for _ in range(10):
                    answers = infer_llm(llm_engine, instruction, None, question, answer_num=50)
                    for answer in answers:
                        # extract code block from answers
                        tmp = re.search(r"```python\n(.*)```", answer, re.DOTALL)
                        # wrong answer format
                        if tmp is None:
                            continue
                        code_block = tmp.group(1)
                        # extract completions according to entry_point
                        tmp = re.findall(r'def ' + re.escape(entry_point) + r'\([^)]*\)[^:\n]*:(.*)', code_block, re.DOTALL)
                        # wrong formats of code block
                        if len(tmp) != 1:
                            continue
                        completions.append(tmp[0])
                        if len(completions) >= 200:
                            break
                    if len(completions) >= 200:
                        break
                    time.sleep(10)
                
                # write to file
                f.write(json.dumps({'task_id': task_id, 'prompt': prompt, 'completions': completions}) + "\n")


class Spec_Generator:
    """
    Generate spec code according to docstring
    """

    @staticmethod
    def prompt(id, dataset):
        """
        Prompt to generate spec code in two functions `preconditions` and `postconditions`
        """
        instruction = "I want you to act as a python programmer. Given a docstring about a python method, you need to write its pre-conditions in one test function \"def preconditions(input)\" and post-conditions in another test function \"def postconditions(input, output):\". You should ensure invalid input or output of the method will raise error in the two test functions."

        exemplars = []
        for i in range(1, 4):
            with open(f"./exemplars/{i}.py", 'r') as f:
                exemplar_codes = f.read()
                exemplar = dict()
                query, answer, _ = re.match(r"(.*)(# Pre-conditions\n.*)(# Test inputs\n.*)", exemplar_codes, re.DOTALL).groups()
                exemplar['query'] = "```python\n" + query + "```\n"
                exemplar['answer'] = "```python\n" + answer + "```\n"

                exemplars.append(exemplar)
        
        question = "```python\n" + dataset[id]['prompt'] + "    pass\n" + "```\n"

        return instruction, exemplars, question


    @staticmethod
    def inference_all(resume_id=None):
        """
        Generate spec code for all samples in human_eval_dataset,
        Each sample with at most 100 spec code samples.
        Then save the results in ../save/{data_type}/specs.jsonl
        """

        # load human_eval_dataset
        with open(f"../data/{data_type}.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f.readlines()]

        # generate original spec code samples
        fd_mode = 'a' if resume_id is not None else 'w'
        resume_id = resume_id if resume_id is not None else 0
        with open(f"../save/{data_type}/specs.jsonl", fd_mode) as f:
            for id in tqdm(range(resume_id, len(dataset))):
                prompt = dataset[id]['prompt']
                task_id = dataset[id]['task_id']
                entry_point = dataset[id]['entry_point']

                # generate prompt
                instruction, exemplars, question = Spec_Generator.prompt(id, dataset)

                spec_code_choices = []
                for _ in range(2):
                    # inference with chatgpt api
                    answers = infer_llm(llm_engine, instruction, exemplars, question, answer_num=60)
                    # extract spec code from answers
                    answers = [re.search(r"```python\n(.*)```", answer, re.DOTALL).group(1) for answer in answers if re.search(r"```python\n(.*)```", answer, re.DOTALL) is not None]
                    spec_code_choices.extend(answers)

                    time.sleep(10)              # generation token limit
                
                f.write(json.dumps({"prompt": prompt, "spec_code_choices": spec_code_choices[:100], "task_id": task_id}) + "\n")
        
                      

class TestCase_Generator:
    """
    Generate test cases according to spec code and docstring
    """    

    @staticmethod
    def prompt(id, dataset):
        """
        Prompt to generate test cases according to only docstring
        """
        instruction = "# Given a docstring, continue to write the following code with 10 valid assertion statements to check the correctness of the function. Provide diverse test cases. \n\n"

        # question = "```python\n" + dataset[id]['prompt'] + "    pass\n" + "assert ```\n"
        question = f"{dataset[id]['prompt']}    pass\n\n# check the correctness of `{dataset[id]['entry_point']}` with 10 different valid assertion statements in the form of \"assert {dataset[id]['entry_point']}(...) == ...\"\nassert "

        return instruction, question

    @staticmethod
    def _test_case_extract(content, entry_point):
        def _truncate(content):
            for identifier in ['\nclass', '\ndef', '\n#', '\nif', '\nprint']:
                if identifier in content:
                    content = content.split(identifier)[0]
            return content.strip()
        
        split_by_assert = [f'assert {part}'.strip() for part in f'assert {content}'.split('assert ') if (entry_point.strip() in part) and len(part.strip()) > 0 and '==' in part]
        truncated_test_cases = [_truncate(i) for i in split_by_assert]
        checked_assertions = [i for i in truncated_test_cases if TestCase_Generator._check_test_case_validation(i)]
        return checked_assertions

    @staticmethod
    def _check_test_case_validation(test_case):
        if len(test_case.strip()) < 1:
            return False
        if 'assert' not in test_case:
            return False
        try:
            multi_line_test_case = test_case.replace("\n", "\n    ")
            assert_in_a_block = f'try:\n    {multi_line_test_case}\nexcept:\n    pass\n'
            compile(assert_in_a_block, '', 'exec')
            return True
        except Exception:
            return False
 
    @staticmethod
    def inference_all(resume_id=None):
        """
        Generate test inputs for all samples in human_eval_dataset,
        Then save the results in ./test_cases.jsonl
        """

        # load human_eval_dataset
        with open(f"../data/{data_type}.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f.readlines()]

        save_data = []
        if resume_id is not None:
            save_data = load_pickle(f"../save/{data_type}/test_cases.pkl")

        for id in tqdm(range(len(dataset))):
            if resume_id is not None and id < resume_id:
                continue

            task_id = dataset[id]['task_id']

            prompt = dataset[id]['prompt']

            entry_point = dataset[id]['entry_point']

            # generate prompt
            instruction, question = TestCase_Generator.prompt(id, dataset)

            while True:
                try:
                    all_valid_test_cases = []
                    for _ in range(4):
                        # inference with chatgpt api
                        rtn = infer_llm_completion(llm_engine, instruction, None, question, answer_num=30, max_tokens=4096)
                        assert len(rtn) >= 1
                        for test_input_code in rtn:
                            for single_assertion in TestCase_Generator._test_case_extract(test_input_code, entry_point):
                                if single_assertion not in all_valid_test_cases:
                                    all_valid_test_cases.append(single_assertion)
                        

                        if len(all_valid_test_cases) >= 500:
                            break
                        time.sleep(10)
                except Exception as e:
                    print(e)

                print(len(all_valid_test_cases))
                save_data.append({"prompt": prompt, "test_cases": all_valid_test_cases, "task_id": task_id})

                try:
                    pickle.dump(save_data, open(f"../save/{data_type}/test_cases_new.pkl", 'wb'))
                    break
                except Exception as e:
                    print(e)
                    save_data = save_data[:-1]
                    continue

            shutil.move(f"../save/{data_type}/test_cases_new.pkl", f"../save/{data_type}/test_cases.pkl")


class SpecTestCase_Verifier:
    """
    Verify the spec code and test cases by executing themselves
    Results are saved in ./spec_test_case_results.jsonl in form of
    {
        "task_id": string of task_id,
        "spec": string of spec_code,
        "result": list of bool, each element indicates whether the corresponding test case is passed
        "fail_reason": string of fail reason
    }
    Note that the spec code is deduplicated in this step
    """

    @staticmethod
    def verify_all():
        """
        Verify all the spec code and test cases in the dataset
        """
        with open(f"../data/{data_type}.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f.readlines()]
        spec_dataset = load_jsonl(f"../save/{data_type}/specs.jsonl")
        test_case_dataset = load_pickle(f"../save/{data_type}/test_cases.pkl")

        test_case_dict = {
            item["task_id"]: item["tc_input_output"][:100] for item in test_case_dataset
        }
        spec_dict = {
            item["task_id"]: item["spec_code_choices"][:50] for item in spec_dataset
        }

        specs = []
        for prompt in dataset:
            task_id = prompt["task_id"]
            
            # determine whether is multi_args
            if data_type == "CodeContests":
                multi_args = False
            else:
                ast_tree = ast.parse(prompt["prompt"])
                for body in ast_tree.body:
                    if isinstance(body, ast.FunctionDef) and body.name == prompt["entry_point"]:
                        multi_args = len(body.args.args) > 1
                        break

            for spec in spec_dict[task_id]:
                specs.append(
                    {
                        "task_id": task_id,
                        "prompt": prompt["prompt"] + "\n    pass\n", # add pass to avoid syntax error
                        "spec": spec,
                        "entry_point": prompt["entry_point"],
                        "multi_args": multi_args,
                    }
                )

        spec_scores = evaluate_spec_with_test_cases(specs, test_case_dict, 0.01)
        dump_pickle(spec_scores, f"../save/{data_type}/spec_testcase_res.pkl")

class CodeTestCase_Verifier:
    """
    Verify the code and test cases by executing themselves
    Results are saved in ./code_test_case_results.jsonl in form of
    {
        "task_id": string of task_id,
        "code": string of code,
        "result": list of bool, each element indicates whether the corresponding test case is passed
        "fail_reason": string of fail reason
    }
    Note that the code is deduplicated in this step
    """

    @staticmethod
    def verify_all():
        with open(f"../data/{data_type}.jsonl", 'r') as f:
            dataset = [json.loads(line) for line in f.readlines()]
        completions = load_jsonl(f"../save/{data_type}/solutions.jsonl")
        test_cases = load_pickle(f"../save/{data_type}/test_cases.pkl")
        
        test_case_dict = {
            item["task_id"]: item["tc_input_output"][:100] for item in test_cases
        }
        solution_dict = {
            item["task_id"]: item["completions"][:200] for item in completions
        }

        solutions = []
        for prompt in dataset:
            task_id = prompt["task_id"]
            
            # determine whether is multi_args
            if data_type == "CodeContests":
                multi_args = False
            else:
                ast_tree = ast.parse(prompt["prompt"])
                for body in ast_tree.body:
                    if isinstance(body, ast.FunctionDef) and body.name == prompt["entry_point"]:
                        multi_args = len(body.args.args) > 1
                        break

            for solution in solution_dict[task_id]:
                solutions.append(
                    {
                        "task_id": task_id,
                        "prompt": prompt["prompt"],
                        "solution": solution,
                        "entry_point": prompt["entry_point"],
                        "multi_args": multi_args,
                    }
                )

        scores = evaluate_with_test_cases(solutions, test_case_dict, 0.01, assert_format=False)
        # Noted that task without spec code will not be included in the scores list
        dump_pickle(scores, f"../save/{data_type}/code_testcase_res.pkl")

class CodeSpec_Verifier:
    """
    Verify the code and spec by executing the code
    Results are saved in ./code_spec_results.jsonl in form of
    {
        "task_id": string of task_id,
        "code": string of code,
        "spec": string of spec_code,
        "result": bool, indicates whether the code passes the spec
        "fail_reason": string of fail reason
    }
    Note that both the code and spec are deduplicated in this step
    """

    @staticmethod
    def verify_all():
        specs = load_jsonl(f"../save/{data_type}/specs.jsonl")
        codes = load_jsonl(f"../save/{data_type}/solutions.jsonl")
        casual_input = load_pickle(f"../save/{data_type}/casual_input.pkl")
        dataset = load_jsonl(f"../data/{data_type}.jsonl")

        spec_dict = {
            item["task_id"]: item["spec_code_choices"][:50] for item in specs
        }
        solution_dict = {
            item["task_id"]: item["completions"][:200] for item in codes
        }
        casual_input_dict = {
            item["task_id"]: [i['input'] for i in item["test_cases"][:30]] for item in casual_input 
        }

        solutions = []
        for prompt in dataset:
            task_id = prompt["task_id"]
            
            # determine whether is multi_args
            if data_type == "CodeContests":
                multi_args = False
            else:
                ast_tree = ast.parse(prompt["prompt"])
                for body in ast_tree.body:
                    if isinstance(body, ast.FunctionDef) and body.name == prompt["entry_point"]:
                        multi_args = len(body.args.args) > 1
                        break
            
            for solution in solution_dict[task_id]:
                solutions.append(
                    {
                        "task_id": task_id,
                        "prompt": prompt["prompt"] if data_type != "CodeContests" else prompt["prompt"] + "\n    pass\n",
                        "solution": solution,
                        "entry_point": prompt["entry_point"],
                        "multi_args": multi_args,
                    }
                )

        result = evaluate_with_specs_and_casual_input(solutions, spec_dict, casual_input_dict, 1)
        dump_pickle(result, f"../save/{data_type}/code_spec_res.pkl")



if __name__ == '__main__':
    # debug = True
    # print("=== Running on debug mode ===")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for data_type in ["human_eval",'mbpp', 'CodeContests']:

        # print("=== Generating solutions ===")
        # Solution_Generator.inference_all()

        # print("=== Generating specs ===")
        # Spec_Generator.inference_all()

        # print("=== Generating test cases ===")
        # TestCase_Generator.inference_all()

        # test_cases = load_pickle(f"../save/{data_type}/test_cases.pkl")
        # # shuffle test cases
        # for item in test_cases:
        #     del item['test_cases']
        #     random.shuffle(item['tc_input_output'])
        # dump_pickle(test_cases, f"../save/{data_type}/test_cases.pkl")

        CodeTestCase_Verifier.verify_all()
        CodeSpec_Verifier.verify_all()
        SpecTestCase_Verifier.verify_all()
