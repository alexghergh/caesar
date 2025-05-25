
"""
Analyze at a particular run 
of multi-turn caesar runs
to get the fast_p score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import json

from utils.score import *

import caesar.analysis.analysis_utils as analysis_utils
from caesar.interface.run_mapping import RUN_MAPPING_LEVEL_1, RUN_MAPPING_LEVEL_2, RUN_MAPPING_LEVEL_3
from caesar.utils import check_result_exists_run_path

PATH_TO_REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

########################################################
#
# We want to get the fast_1 score of the best solution up to turn k
# we want to understand the fast_1 score of the best solution up to turn k

# For each the problem
# 1. grab all the eval results up to turn k
# 2. get the best

# and then compute fast_p score across all the problems

########################################################

LOG_DIR_PREFIX = "/matx/u/simonguo/kernel_multi_turn"


def grab_eval_results_at_k(run_path: str, 
                              problem_id: int, 
                              sample_id: int, 
                              k: int) -> float:
    """
    We need the path to the log.json
    """
    # if not check_result_exists_run_path(run_path, problem_id, sample_id):
    #     print(f"WARNING: run_path {run_path} does not exist for problem {problem_id} sample {sample_id}")
    #     return -1 # TODO CHECK THIS
    
    log_path = os.path.join(run_path, f"problem_{problem_id}", f"sample_{sample_id}", "log.json")
    log_json = analysis_utils.load_run_data(log_path)

    # check the problem_id and sample_id matches
    assert str(log_json["metadata"]["problem_id"]) == str(problem_id), f"Problem ID mismatch: {log_json['metadata']['problem_id']} != {problem_id}"
    assert str(log_json["metadata"]["sample_id"]) == str(sample_id), f"Sample ID mismatch: {log_json['metadata']['sample_id']} != {sample_id}"
    
    # get the eval results atturn k
    turn_num = k
    if str(turn_num) not in log_json:
        print(f"WARNING: turn {turn_num} not found in log for problem {problem_id} sample {sample_id}")
        return -1
    
    else:
        if "eval_result" not in log_json[str(turn_num)]:
            print(f"WARNING: eval_result not found in log for turn {turn_num} for problem {problem_id} sample {sample_id}")
            return -1
        else:
            runtime = log_json[str(turn_num)]["eval_result"]["runtime"]
            return runtime
            


def grab_eval_results_up_to_k(run_path: str, 
                              problem_id: int, 
                              sample_id: int, 
                              k: int) -> list[float]:
    """
    We need the path to the log.json
    """
    # if not check_result_exists_run_path(run_path, problem_id, sample_id):
    #     print(f"WARNING: Run {run_path} DID NOT FINISH for problem {problem_id} sample {sample_id}")
    #     return [-1]*k # TODO CHECK THIS

    log_path = os.path.join(run_path, f"problem_{problem_id}", f"sample_{sample_id}", "log.json")
    log_json = analysis_utils.load_run_data(log_path)

    # check the problem_id and sample_id matches
    assert str(log_json["metadata"]["problem_id"]) == str(problem_id), f"Problem ID mismatch: {log_json['metadata']['problem_id']} != {problem_id}"
    assert str(log_json["metadata"]["sample_id"]) == str(sample_id), f"Sample ID mismatch: {log_json['metadata']['sample_id']} != {sample_id}"

    # just clean up eval result
    # we get a eval_time (list) of length k
    
    runtime_up_to_k = []
    
    # get the eval results up to turn k
    for turn_num in range(1, k + 1):
        if str(turn_num) not in log_json:
            print(f"WARNING: turn {turn_num} not found in log for problem {problem_id} sample {sample_id}")
            runtime_up_to_k.append(-1)
        
        else:
            if "eval_result" not in log_json[str(turn_num)]:
                print(f"WARNING: eval_result not found in log for turn {turn_num} for problem {problem_id} sample {sample_id}")
                runtime_up_to_k.append(-1)
            else:
                runtime = log_json[str(turn_num)]["eval_result"]["runtime"]
                runtime_up_to_k.append(runtime)
                       
    return runtime_up_to_k

def get_best_solution(sol: list[float]) -> float | None:
    """
    For a single problem, we find the best solution up to turn k
    # if there is multiple thigs that are correct, we pick the best one
    # if there is only one correct, we use that one
    # if there is no correct, no correct
    """
    m = None
    for s in sol:
        if s == -1: # skip if it is not correct
            continue
        if m is None:
            m = s
        else:
            m = min(m,s)
    return m  


def get_overall_runtime(run_path: str, 
                        turn_k: int,
                        num_problems: int = 100):
    """
    For a single run, we compute the runtime across all the problems
    """

    # do this for all the problems
    problem_ids = range(1, num_problems + 1) # for level 1

    overall_runtime = [] # this should be 
    for problem_id in problem_ids:
        # 
        curr_runtime_up_to_k = grab_eval_results_up_to_k(
            run_path=run_path,    
            problem_id=problem_id, 
            sample_id=0, 
            k=turn_k)
        
        assert len(curr_runtime_up_to_k) == turn_k, f"Expected {turn_k} runtime results for problem {problem_id}, got {len(curr_runtime_up_to_k)}"

        best_runtime = get_best_solution(curr_runtime_up_to_k)

        if best_runtime is None: # this means it is not correct
            # print(f"WARNING: No correct solutions found for problem {problem_id}")
            overall_runtime.append(None) # this means no correct solutions were found
        else:
            overall_runtime.append(best_runtime)
        
    return overall_runtime

def compute_fast_p_score(overall_runtime: list[float | None],
                         baseline_torch_time_filepath: str,
                         level: int,
                         num_problems: int = 100,
                         p: float = 1.0) -> float:
    """
    Compute the fast_p score
    """
    # get the baseline time array
    baseline_time_array = []
    for problem_id in range(1, num_problems + 1):
        curr_problem_baseline_time = analysis_utils.fetch_baseline_time_by_problem_id(level=level,
                                                                                      problem_id=problem_id,
                                                                                      baseline_time_filepath=baseline_torch_time_filepath).get("mean", None)
        baseline_time_array.append(curr_problem_baseline_time)
    
    return fastp(is_correct=np.array([x is not None for x in overall_runtime]),
        baseline_speed=np.array(baseline_time_array),
        actual_speed=np.array(overall_runtime),
        n=num_problems,
        p=p)

def compute_fast_1_score(overall_runtime: list[float | None],
                         baseline_torch_time_filepath: str,
                         level: int,
                         num_problems: int = 100) -> float:
    return compute_fast_p_score(overall_runtime, baseline_torch_time_filepath, level, num_problems, p=1.0)
    


def compute_fast_p_for_run_with_turn_k(run_path: str,
                                       turn_k: int,
                                       level: int,
                                       baseline_torch_time_filepath: str,
                                       num_problems: int = 100,
                                       p: float = 1.0) -> float:
    """
    Compute the fast_p score for a single run with a given turn k
    """
    # get the overall runtime
    overall_runtime = get_overall_runtime(run_path=run_path, 
                                          turn_k=turn_k, 
                                          num_problems=num_problems)
    
    print(overall_runtime)
    fast_0 = sum([1 for x in overall_runtime if x is not None]) / len(overall_runtime)
    return fast_0
    # return compute_fast_p_score(
    #     overall_runtime=overall_runtime,
    #     baseline_torch_time_filepath=baseline_torch_time_filepath,
    #     level=level,
    #     num_problems=num_problems,
    #     p=p)


def get_fast_0_score_for_run_group(run_path: str,
                                   baseline_torch_time_filepath: str,
                                   level: int,
                                   num_problems: int,
                                   k_s: list[int]) -> dict:
    """
    Get the fast_0 score for a run group
    """
    fast_0_scores = {}
    for k in k_s:
        fast_0_scores[k] = compute_fast_p_for_run_with_turn_k(run_path=run_path,
                                                              turn_k=k,
                                                              level=level,
                                                              baseline_torch_time_filepath=baseline_torch_time_filepath,
                                                              num_problems=num_problems,
                                                              p=0.0)

    return fast_0_scores


def main():
    k_s = [10]

    baseline_torch_time_filepath = os.path.join(PATH_TO_REPO_DIR, "KernelBenchInternal", "results", "timing", "L40S_matx3", "baseline_time_torch.json")

    for run_group, level, num_problems in zip([RUN_MAPPING_LEVEL_1, RUN_MAPPING_LEVEL_2, RUN_MAPPING_LEVEL_3], [1,2,3], [100, 100, 50]):
        # grab only eval_result_last_only
        deepseek_v3_run = run_group["eval_result_last_only"]["deepseek-v3"]
        deepseek_v3_run_path = os.path.join(LOG_DIR_PREFIX, deepseek_v3_run["run_group"], deepseek_v3_run["run_name"])
        llama_run = run_group["eval_result_last_only"]["llama-3.1-70b-inst"]
        llama_run_path = os.path.join(LOG_DIR_PREFIX, llama_run["run_group"], llama_run["run_name"])
        
        fast_0_scores_deepseek_v3 = get_fast_0_score_for_run_group(run_path=deepseek_v3_run_path,
                                        baseline_torch_time_filepath=baseline_torch_time_filepath,
                                        level=level,
                                        num_problems=num_problems,
                                        k_s=k_s)    
        

        print(f"Fast 0 scores for deepseek-v3 on level {level} run {deepseek_v3_run['run_name']}: {fast_0_scores_deepseek_v3}")

        fast_0_scores_llama = get_fast_0_score_for_run_group(run_path=llama_run_path,
                                        baseline_torch_time_filepath=baseline_torch_time_filepath,
                                        level=level,
                                        num_problems=num_problems,
                                        k_s=k_s)       
        
        print(f"Fast 0 scores for llama-3.1-70b-inst on level {level} run {llama_run['run_name']}: {fast_0_scores_llama}")

    # # analyze one particular run]
    # # run_group = "level1_eval_result_profiler_last_only_deepseek"
    # # run_name = "run_v0_deepseek_r1_turn"
    
    # # run_group = "level2_reflection_all_prev_deepseek"
    # # run_name = "run_deepseek_turns_10"

    # # run_group = "level3_reflection_last_only_deepseek"
    # # run_name = "run_deepseek_turns_10"

    # run_group = "level3_reflection_all_prev_deepseek"
    # run_name = "run_v0_deepseek_r1_turn"

    # level = 3
    # num_problems = 50

    # run_path = os.path.join(LOG_DIR_PREFIX, run_group, run_name)
    # baseline_torch_time_filepath = os.path.join(PATH_TO_REPO_DIR, "KernelBenchInternal", "results", "timing", "L40S_matx3", "baseline_time_torch.json")


    # print(compute_fast_p_for_run_with_turn_k(run_path=run_path, 
    #                                          turn_k=10, 
    #                                          level=level,
    #                                          baseline_torch_time_filepath=baseline_torch_time_filepath,
    #                                          num_problems=num_problems,
    #                                          p=1.0))
   
    # # DEBUG
    # # overall_runtime = get_overall_runtime(run_path, 10)
    # # print(overall_runtime)
  
    # # # calculate the fast_1 score 
    # # baseline_torch_time_filepath = os.path.join(PATH_TO_REPO_DIR, "KernelBenchInternal", "results", "timing", "L40S_matx3", "baseline_time_torch.json")
    # # curr_run_fast_1_score = compute_fast_1_score(overall_runtime, baseline_torch_time_filepath)
    # # print(f"Fast 1 score: {curr_run_fast_1_score}")
    # # # runtime_up_to_k = grab_eval_results_up_to_k(
    # # #     run_path=run_path,    
    # # #     problem_id=1, 
    # # #     sample_id=0, 
    # # #     k=10)
    # # # construct the baseline array

    
    # # best_runtime = get_best_solution(runtime_up_to_k)
    
    # # print("Turn runtimes:", runtime_up_to_k)
    # # print("Best runtime:", best_runtime)
    
    # # run_data = get_run_data(run_group, run_name, dataset)


if __name__ == "__main__":
    main()