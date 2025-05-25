

from dataclasses import dataclass
from typing import Dict
import pydra
from pydra import Config, REQUIRED
import shutil
import torch
import os, json
from monkeys.problems.kernelbench_gen_level_1 import DATASET as KERNELBENCH_LEVEL_1_DATASET, SUBSET_DATASET as KERNELBENCH_LEVEL_1_SUBSET_DATASET   
from monkeys.problems.kernelbench_gen_level_2 import DATASET as KERNELBENCH_LEVEL_2_DATASET, SUBSET_DATASET as KERNELBENCH_LEVEL_2_SUBSET_DATASET
from monkeys.problems.kernelbench_gen_level_3 import DATASET as KERNELBENCH_LEVEL_3_DATASET, SUBSET_DATASET as KERNELBENCH_LEVEL_3_SUBSET_DATASET
from eval import evaluate_single_sample_src
# from utils import make_json_serializable

from KernelBenchInternal.src.utils import read_file
from KernelBenchInternal.src import utils as kernel_utils

dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET
}


"""
Simple file to clean up log as well
As well as adding evals turn by turn if we need to

1. If one of the turn info is missing, we delete
2. If turn is incomplete, we delete
dict empty
keys  remove 
run eval reuslt there is kernel

Usage: 
- python3 clean_log.py run_group="level3_reflection_all_prev_deepseek" remove_file=True

SHOULD REALLY DOUBLE CHECK THIS FILE
"""

class CaesarRunConfig(Config):
    def __init__(self):
        # run
        # dataset

        self.num_workers = 1 # let's do one 
        # self.num_gpu_workers for later, assume we get one gpu for each yet

        # Eval Speciifc  
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 300
        self.verbose = False
        self.measure_performance = True

        # Logging
        self.log_dir = "/matx/u/simonguo/kernel_multi_turn/"
        self.build_dir_prefix = "/matx/u/simonguo/kernel_eval_build/" 
        self.gpu_arch = ["Ada"] # build for L40s Ada Lovelace architecture

        self.run_group = REQUIRED
        self.specific_run_name = ""
        # DEFAULT clean all runs

        # testing only 1 sample
        # self.testing = False
        self.debug = False
        self.remove_file = True

        self.problem_id = 42
        self.sample_id = 0



def read_clean_log(dir: str | os.PathLike, remove_file: bool = False) -> bool:
    """
    Check log logic

    Return whether if it was corrupted and removed
    """
    log_file = os.path.join(dir, "log.json")
    if not os.path.exists(log_file):
        print(f"Log file not found in {dir}")
        if remove_file:
            print(f"Removing {dir}")
            shutil.rmtree(dir)
        return True

    with open(log_file, "r") as f:
        curr_log = json.load(f)


    is_corrupted = False
    max_turn = curr_log["metadata"]["num_rounds"]
    
    for turn in curr_log.keys():
        if not turn.isdigit():
            continue
        
        # for the current turn, check if content exists
        if curr_log[str(turn)] == {}:
            print(f"Turn {turn} is empty, removing")
            is_corrupted = True
            break

        # This is very aggresive deleting! 
        # another way is to invoke GPU to just add eval_result 
        # check if all fields of the curr_turn entry has been filled

        
        # Check required fields
        required_fields = ["context", "model_response", "kernel_code"]
        for field in required_fields:
            if field not in curr_log[str(turn)] or curr_log[str(turn)][field] == "":
                print(f"Required field {field} is missing or empty, removing")
                is_corrupted = True
                break
        
        # Check that at least one of feedback or eval_result exists
        if not is_corrupted:
            if ("feedback" not in curr_log[str(turn)] or curr_log[str(turn)]["feedback"] == "") and \
               ("eval_result" not in curr_log[str(turn)] or curr_log[str(turn)]["eval_result"] == ""):
                print("Both feedback and eval_result are missing or empty, removing")
                is_corrupted = True

    
    if is_corrupted:
        print(f"Log corrupted in {dir}")
        if remove_file:
            print(f"Removing {dir}")
            shutil.rmtree(dir)
    
    return is_corrupted


@pydra.main(base=CaesarRunConfig)
def main(config: CaesarRunConfig):
    """
    ONGOING
    Overal logic:
    1. read existing log
    2. for kernel at each turn, eval them
    3. write the results back
    """
    print("Running with config", config)

    if config.remove_file:
        confirm = input(f"[DANGER] Are you sure you want to remove corrupted files? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborting removal")
            return
    total_corrupted_and_processed = 0
    num_corrupted_and_processed = 0
    
    run_group_dir = os.path.join(config.log_dir, config.run_group)
    # Get list of run directories to process
    run_dirs = [os.path.join(run_group_dir, config.specific_run_name)] if config.specific_run_name else [os.path.join(run_group_dir, run_name) for run_name in os.listdir(run_group_dir)]

    for run_dir in run_dirs:
        print(f"Cleaning run {run_dir}")
        
        for problem_id in os.listdir(run_dir):
            for sample_id in os.listdir(os.path.join(run_dir, problem_id)):
                curr_file_corrupted = read_clean_log(os.path.join(run_dir, problem_id, sample_id), remove_file=config.remove_file)
                if curr_file_corrupted:
                    num_corrupted_and_processed += 1

        print(f"Processed {num_corrupted_and_processed} corrupted files in {run_dir}")            
        total_corrupted_and_processed += num_corrupted_and_processed

    print(f"Total corrupted files: {total_corrupted_and_processed} Found and Removed")



    # # TODO: problem should be in the log
    # problems = dataset_name_to_dataset[config.dataset_name]
    # problem = problems[config.problem_id] # this should be a path
    
    # # Fetch Ref Arch Src
    # # assume each problem is a path 
    # problem_path_prefix = "../" # to KernelBench directory
    # problem_path = os.path.join(problem_path_prefix, problem)
    
    # ref_arch_src = read_file(problem_path)

    # # TODO: this need to be updated
    # log_file = os.path.join(config.log_dir, f"caesar_log_{config.problem_id}.json")
    # with open(log_file, "r") as f:
    #     log = json.load(f)
    
    # max_turn = max(int(turn) for turn in log.keys())
    # print(f"Max turn found in log: {max_turn}")

    # kernel_utils.set_gpu_arch(config.gpu_arch)

    # eval_results = []

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # for turn in range(1, max_turn + 1):
    #     print(f"Processing turn {turn}")
        
    #     # import pdb; pdb.set_trace()
    #     kernel_src = log[str(turn)]["kernel_code"]
        
    #     # TO UPDATE:
    #     build_dir = os.path.join(config.build_dir_prefix, f"{config.problem_id}", f"turn_{turn}")

    #     eval_result = evaluate_single_sample_src(ref_arch_src=ref_arch_src, kernel_src=kernel_src, configs=config, build_dir=build_dir, device=device)

    #     eval_result = make_json_serializable(eval_result)

    #     print(f"Eval result for turn {turn}: {eval_result}")
        
        
    #     eval_results.append((turn, eval_result))
        

    # # write to a file
    # with open(os.path.join(config.log_dir, f"eval_results_{config.problem_id}.json"), "w") as f:
        
    #     json.dump(eval_results, f)

    

if __name__ == "__main__":
    main()
