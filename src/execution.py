# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Mostly copied from CodeT (https://github.com/microsoft/CodeT/tree/main/CodeT)

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from collections import defaultdict
import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
import logging

from _execution import check_correctness, check_correctness_with_test_cases, check_spec_with_test_cases, check_with_specs_and_test_cases, check_with_specs_and_casual_inputs

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def evaluate_with_test_code(
    samples,
    timeout
):
    logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    with ProcessPoolExecutor(40) as executor:

        futures = []
        existed_completion = defaultdict(set)
        results = defaultdict(defaultdict)

        for sample in samples:
            task_id = sample["task_id"]
            prompt = sample['prompt']
            test = sample['test']
            entry_point = sample['entry_point']
            completion = sample["completion"]
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            args = (task_id, prompt, completion, test, entry_point, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        logger.info(f'{len(futures)} execution requests are submitted')
        
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results[result["task_id"]][result["completion"]] = result

    logger.info('execution finished! start parsing results')
    samples_with_result = []
    for sample in samples:
        task_id = sample["task_id"]
        completion = sample["completion"]
        result = results[task_id][completion]
        sample["result"] = result["result"]
        sample["passed"] = result["passed"]
        samples_with_result.append(sample)

    assert len(samples_with_result) == len(samples), "Some problems are not attempted."

    return samples_with_result

def evaluate_with_test_cases(
    solutions,
    test_cases_dict,
    timeout,
    assert_format=False
):
    logger.info(f'Start evaluation with test cases, timeout={timeout}')
    # Check the generated solutions against test suites.
    with ProcessPoolExecutor(25) as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in solutions:
            task_id = solution['task_id']
            prompt = solution['prompt']
            completion = solution['solution']
            multi_args = solution['multi_args']
            entry_point = solution['entry_point']
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            task_test_cases = test_cases_dict[task_id]
            args = (task_id, prompt, completion, task_test_cases, entry_point, timeout, multi_args, assert_format)
            # check_correctness_with_test_cases(*args)
            future = executor.submit(check_correctness_with_test_cases, *args)
            futures.append(future)

        logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)

    logger.info('execution finished!')
    return results_list


def evaluate_spec_with_test_cases(
    specs,
    test_cases_dict,
    timeout
):
    logger.info(f'Start evaluation with test cases, timeout={timeout}')
    # Check the generated specs against test suites.
    with ProcessPoolExecutor(25) as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in specs:
            task_id = solution['task_id']
            prompt = solution['prompt']
            spec = solution['spec']
            multi_args = solution['multi_args']
            if spec in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(spec)
            task_test_cases = test_cases_dict[task_id]
            args = (task_id, prompt, spec, task_test_cases, timeout, multi_args)
            # check_spec_with_test_cases(*args)
            future = executor.submit(check_spec_with_test_cases, *args)
            futures.append(future)

        logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)

    logger.info('execution finished!')
    return results_list

def evaluate_with_specs_and_casual_input(
    solutions,
    specs_dict,
    casual_input_dict,
    timeout
):
    logger.info(f'Start evaluation with specs and casual input, timeout={timeout}')
    # Check the generated specs against test suites.
    with ProcessPoolExecutor(40) as executor:
    # with multiprocessing.Pool(20) as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in solutions:
            task_id = solution['task_id']
            prompt = solution['prompt']
            completion = solution['solution']
            multi_args = solution['multi_args']
            entry_point = solution['entry_point']
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)

            task_specs = specs_dict[task_id]  # already been deduplicated
            casual_inputs = casual_input_dict[task_id]
            if not casual_inputs:
                continue
            for spec in task_specs:
                args = (task_id, prompt, completion, spec, casual_inputs, entry_point, timeout, multi_args)
                # check_with_specs_and_casual_inputs(*args)
                future = executor.submit(check_with_specs_and_casual_inputs, *args)
                # future = executor.apply_async(check_with_specs_and_casual_inputs, args=args)
                futures.append(future)
                logger.info(f'{len(futures)} execution requests are submitted')

        logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)
        # for idx, future in enumerate(futures):
        #     result = future.get()
        #     results_list.append(result)
        #     logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))

    logger.info('execution finished!')
    return results_list


def evaluate_with_specs_and_test_input(
    solutions,
    specs_dict,
    test_cases_dict,
    timeout
):
    logger.info(f'Start evaluation with test cases, timeout={timeout}')
    # Check the generated specs against test suites.
    with ProcessPoolExecutor(40) as executor:
        futures = []
        results_list = []
        existed_completion = defaultdict(set)

        for solution in solutions:
            task_id = solution['task_id']
            prompt = solution['prompt']
            completion = solution['solution']
            multi_args = solution['multi_args']
            entry_point = solution['entry_point']
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            task_specs = specs_dict[task_id]
            task_test_cases = test_cases_dict[task_id]
            if not task_test_cases:
                continue
            
            for spec in task_specs:
                args = (task_id, prompt, completion, spec, task_test_cases, entry_point, timeout, multi_args)
                future = executor.submit(check_with_specs_and_test_cases, *args)
                futures.append(future)

        logger.info(f'{len(futures)} execution requests are submitted')
        for idx, future in enumerate(as_completed(futures)):
            logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
            result = future.result()
            results_list.append(result)

    logger.info('execution finished!')
    return results_list
