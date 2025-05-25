

from dataclasses import dataclass
from typing import Dict
import pydra
from pydra import Config, REQUIRED
import torch
import os, json
from monkeys.problems.kernelbench_gen_level_1 import DATASET as KERNELBENCH_LEVEL_1_DATASET, SUBSET_DATASET as KERNELBENCH_LEVEL_1_SUBSET_DATASET   
from monkeys.problems.kernelbench_gen_level_2 import DATASET as KERNELBENCH_LEVEL_2_DATASET # SUBSET_DATASET as KERNELBENCH_LEVEL_2_SUBSET_DATASET
from monkeys.problems.kernelbench_gen_level_3 import DATASET as KERNELBENCH_LEVEL_3_DATASET # SUBSET_DATASET as KERNELBENCH_LEVEL_3_SUBSET_DATASET
from eval import evaluate_single_sample_src
from utils import make_json_serializable

from KernelBenchInternal.src.utils import read_file
from KernelBenchInternal.src import utils as kernel_utils

dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET
}


"""
Simple Script to eval turn by turn
"""

class CaesarRunConfig(Config):
    def __init__(self):
        # run
        self.run_name = REQUIRED
        # dataset
        self.dataset_name = "KernelBench/level1"
        self.num_samples = 1

        # multi-turn stuff
        self.max_k = 10
        self.context_strategy = ["reflection"]

        self.num_workers = 1 # let's do one 
        # self.num_gpu_workers for later, assume we get one gpu for each yet

        # Eval Speciifc  
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 300
        self.verbose = False
        self.measure_performance = True

        # Logging
        self.log_dir = "./logs"
        self.build_dir_prefix = "/matx/u/simonguo/kernel_eval_build/" 
        self.gpu_arch = ["Ada"] # build for L40s Ada Lovelace architecture

        self.mock = True
        # testing only 1 sample
        # self.testing = False
        self.debug = False

        self.problem_id = 42
        self.sample_id = 0

@pydra.main(base=CaesarRunConfig)
def main(config: CaesarRunConfig):
    """
    Overal logic:
    1. read existing log
    2. for kernel at each turn, eval them
    3. write the results back
    
    # no device parallelism right now just very simple sequentially process the kernels

    PROBLEM TODO: somehow no cache hit for now!
    """
    print("Running with config", config)

    # TODO: problem should be in the log
    problems = dataset_name_to_dataset[config.dataset_name]
    problem = problems[config.problem_id] # this should be a path
    
    # Fetch Ref Arch Src
    # assume each problem is a path 
    problem_path_prefix = "../" # to KernelBench directory
    problem_path = os.path.join(problem_path_prefix, problem)
    
    ref_arch_src = read_file(problem_path)

    # TODO: this need to be updated
    log_file = os.path.join(config.log_dir, f"caesar_log_{config.problem_id}.json")
    with open(log_file, "r") as f:
        log = json.load(f)
    
    max_turn = max(int(turn) for turn in log.keys())
    print(f"Max turn found in log: {max_turn}")

    kernel_utils.set_gpu_arch(config.gpu_arch)

    eval_results = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for turn in range(1, max_turn + 1):
        print(f"Processing turn {turn}")
        
        # import pdb; pdb.set_trace()
        kernel_src = log[str(turn)]["kernel_code"]
        
        # TO UPDATE:
        build_dir = os.path.join(config.build_dir_prefix, f"{config.problem_id}", f"turn_{turn}")

        eval_result = evaluate_single_sample_src(ref_arch_src=ref_arch_src, kernel_src=kernel_src, configs=config, build_dir=build_dir, device=device)

        eval_result = make_json_serializable(eval_result)

        print(f"Eval result for turn {turn}: {eval_result}")
        
        
        eval_results.append((turn, eval_result))
        

    # write to a file
    with open(os.path.join(config.log_dir, f"eval_results_{config.problem_id}.json"), "w") as f:
        
        json.dump(eval_results, f)

    

if __name__ == "__main__":
    main()
