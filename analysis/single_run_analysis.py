import os
import sys

import numpy as np

from KernelBenchInternal.score import fastp
from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_DATASET,
    KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    KERNELBENCH_LEVEL_1_RANDOM_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_DATASET,
    KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_RANDOM_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_DATASET,
    KERNELBENCH_LEVEL_3_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_RANDOM_SUBSET_DATASET,
    KERNELBENCH_LEVELS_12_REPRESENTATIVE_DATASET,
    CUDAFORGE_SUBSET,
)

# get root caesar directory (i.e., the parent of 'analysis')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# add sys paths (needed for python import discovery)
sys.path.append(ROOT_DIR)

from utils import (
    get_turn_input_tokens,
    get_turn_output_tokens,
    load_json_data,
    fetch_baseline_time_by_problem_id,
)


# timing result to compare against
TIMING_BASELINE = "H100_tsubame"

BASE_LOG_DIR = os.path.join(ROOT_DIR, "caesar_log_dir")
KERNEL_BENCH_PATH = os.path.join(ROOT_DIR, "..", "KernelBench")

dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET,
    "KernelBench/level1-subset": KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    "KernelBench/level2-subset": KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    "KernelBench/level3-subset": KERNELBENCH_LEVEL_3_SUBSET_DATASET,
    "KernelBench/level1-random": KERNELBENCH_LEVEL_1_RANDOM_SUBSET_DATASET,
    "KernelBench/level2-random": KERNELBENCH_LEVEL_2_RANDOM_SUBSET_DATASET,
    "KernelBench/level3-random": KERNELBENCH_LEVEL_3_RANDOM_SUBSET_DATASET,

    "KernelBench/levels12-subset": KERNELBENCH_LEVELS_12_REPRESENTATIVE_DATASET,

    "KernelBench/cudaforge-subset": CUDAFORGE_SUBSET,

    # debug
    "KernelBench/level1-test": [
        os.path.join(KERNEL_BENCH_PATH, "KernelBench", "level1", "23_Softmax.py")
    ],
}


def get_eval_results_at_k(run_path: str,
                          problem_id: int,
                          sample_id: int,
                          k: int) -> float:
    log_path = os.path.join(run_path, f"problem_{problem_id}", f"sample_{sample_id}", "log.json")
    log_json = load_json_data(log_path)

    # get the eval results at turn k
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


def get_eval_results_up_to_k(run_path: str,
                             problem_id: int,
                             sample_id: int,
                             k: int) -> list[float]:
    log_path = os.path.join(run_path, f"problem_{problem_id}", f"sample_{sample_id}", "log.json")
    log_json = load_json_data(log_path)

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
                runtime = log_json[str(turn_num)]["eval_result"].get("runtime", -1.0)
                runtime_up_to_k.append(runtime)

    return runtime_up_to_k


def get_best_solution(solutions: list[float]) -> float | None:
    m = None
    for s in solutions:
        if s == -1: # skip if it is not correct
            continue
        if m is None:
            m = s
        else:
            m = min(m, s)
    return m


def get_overall_best_runtime_for_problem(run_path: str,
                                         max_k: int,
                                         num_samples: int,
                                         problem_ids: list[int]) -> list[float | None]:
    overall_runtimes = []
    for problem_id in problem_ids:
        best_runtime = float('inf')
        for sample_id in range(num_samples):
            curr_runtimes_up_to_k = get_eval_results_up_to_k(
                run_path = run_path,
                problem_id=problem_id,
                sample_id=sample_id,
                k=max_k,
            )

            best_curr_runtime = get_best_solution(curr_runtimes_up_to_k)
            if best_curr_runtime is not None and best_curr_runtime < best_runtime:
                best_runtime = best_curr_runtime

        if best_runtime is None:
            overall_runtimes.append(None) # no correct solutions found
        else:
            overall_runtimes.append(best_runtime)

    return overall_runtimes


def get_overall_mean_runtime_for_problem(run_path: str,
                                         max_k: int,
                                         num_samples: int,
                                         problem_ids: list[int]) -> list[float | None]:
    """
    Results for mean@k results, meaning that the mean runtime is returned,
    instead of the best runtime. This somewhat tells whether the model got lucky
    with a good kernel or if it consistently generated good kernels.
    """
    overall_runtimes = []
    for problem_id in problem_ids:
        runtimes = []
        count = 0
        for sample_id in range(num_samples):
            curr_runtimes_up_to_k = get_eval_results_up_to_k(
                run_path = run_path,
                problem_id=problem_id,
                sample_id=sample_id,
                k=max_k,
            )

            for s in curr_runtimes_up_to_k:
                if s == -1:
                    continue
                runtimes.append(s)
                count += 1

        if len(runtimes) == 0:
            overall_runtimes.append(None) # no correct solutions found
        else:
            overall_runtimes.append(sum(runtimes) / count)

    return overall_runtimes


def compute_fast_p_score(overall_runtime: list[float | None],
                         baseline_torch_time_filepath: str,
                         level: int,
                         problem_ids: list[int],
                         p: float = 1.0) -> float:
    # get the baseline time array
    baseline_time_array = []
    for problem_id in problem_ids:
        curr_problem_baseline_time = fetch_baseline_time_by_problem_id(level=level,
                                                                       problem_id=problem_id,
                                                                       baseline_time_filepath=baseline_torch_time_filepath).get("mean", None)
        baseline_time_array.append(curr_problem_baseline_time)

    return fastp(is_correct=np.array([x is not None for x in overall_runtime]),
                 baseline_speed=np.array(baseline_time_array),
                 actual_speed=np.array(overall_runtime),
                 n=len(problem_ids),
                 p=p)


def compute_best_fast_p_for_run(run_path: str,
                                max_k: int,
                                num_samples: int,
                                level: int,
                                baseline_torch_time_filepath: str,
                                problem_ids: list[int],
                                p: float = 1.0) -> float:
    # get the overall runtime
    overall_runtime = get_overall_best_runtime_for_problem(run_path=run_path,
                                                           max_k=max_k,
                                                           num_samples=num_samples,
                                                           problem_ids=problem_ids)
    return compute_fast_p_score(
        overall_runtime=overall_runtime,
        baseline_torch_time_filepath=baseline_torch_time_filepath,
        level=level,
        problem_ids=problem_ids,
        p=p)


def compute_mean_fast_p_for_run(run_path: str,
                                max_k: int,
                                num_samples: int,
                                level: int,
                                baseline_torch_time_filepath: str,
                                problem_ids: list[int],
                                p: float = 1.0) -> float:
    # get the overall runtime
    overall_runtime = get_overall_mean_runtime_for_problem(run_path=run_path,
                                                           max_k=max_k,
                                                           num_samples=num_samples,
                                                           problem_ids=problem_ids)
    return compute_fast_p_score(
        overall_runtime=overall_runtime,
        baseline_torch_time_filepath=baseline_torch_time_filepath,
        level=level,
        problem_ids=problem_ids,
        p=p)


def compute_input_tokens(run_path: str,
                         max_k: int,
                         num_samples: int,
                         problem_ids: list[int]) -> int:
    count = 0
    for problem_id in problem_ids:
        for sample_id in range(num_samples):
            log_data = load_json_data(os.path.join(run_path,
                                                   f"problem_{problem_id}",
                                                   f"sample_{sample_id}",
                                                   "log.json"))
            for idx, turn_data in log_data.items():
                if int(idx) > max_k:
                    break
                count += get_turn_input_tokens(turn_data)
    return count


def compute_output_tokens(run_path: str,
                          max_k: int,
                          num_samples: int,
                          problem_ids: list[int]) -> int:
    count = 0
    for problem_id in problem_ids:
        for sample_id in range(num_samples):
            log_data = load_json_data(os.path.join(run_path,
                                                   f"problem_{problem_id}",
                                                   f"sample_{sample_id}",
                                                   "log.json"))
            for idx, turn_data in log_data.items():
                if int(idx) > max_k:
                    break
                count += get_turn_output_tokens(turn_data)
    return count


def main():
    run_group = "claude-4-5-haiku-reasoning"
    run_name = "level1-subset-max_k-8-samples-4"

    level = 1
    dataset = KernelBenchDataset(dataset=dataset_name_to_dataset["KernelBench/level1-subset"])

    run_path = os.path.join(BASE_LOG_DIR, run_group, run_name)
    baseline_torch_time_filepath = os.path.join(
        ROOT_DIR,
        "..",
        "KernelBench",
        "results",
        "timing",
        TIMING_BASELINE,
        "baseline_time_torch.json",
    )

    fastp = 1
    max_k = 8 # modify this to get best/mean@k, where k doesn't have to be max_k
    samples = 4

    ## There's a number of interesting statistics that we want
    ## 1. fast-p scores (with best kernel - best@k)
    ## 2. fast-p scores (with mean runtime - mean@k)
    ## 3. for a given p, calculate fast-p trajectory, given turns budgets
    ## 4. number of used tokens (input/output)

    print(f"Run: {run_group}/{run_name}")
    print(f"Results@k, with k={max_k}, samples={samples}")

    # 1. best fast-p scores
    print("=== Best@k ===")
    for p in [0, 0.5, 0.8, 1, 1.5, 2]:
        print(f"Fast-{p}: ", compute_best_fast_p_for_run(run_path=run_path,
                                                         max_k=max_k,
                                                         num_samples=samples,
                                                         level=level,
                                                         baseline_torch_time_filepath=baseline_torch_time_filepath,
                                                         problem_ids=dataset.problem_ids,
                                                         p=p))
    # 2. mean fast-p scores
    print("=== Mean@k ===")
    for p in [0, 0.5, 0.8, 1, 1.5, 2]:
        print(f"Fast-{p}: ", compute_mean_fast_p_for_run(run_path=run_path,
                                                         max_k=max_k,
                                                         num_samples=samples,
                                                         level=level,
                                                         baseline_torch_time_filepath=baseline_torch_time_filepath,
                                                         problem_ids=dataset.problem_ids,
                                                         p=p))

    # 3. for a given p, calculate fast-p trajectory, given turns budgets
    print("===fast-p trajectory@p ===")
    for turns in range(1, max_k + 1):
        print(f"Fast-{fastp} with {turns} turns: ", compute_best_fast_p_for_run(run_path=run_path,
                                                                                max_k=turns,
                                                                                num_samples=samples,
                                                                                level=level,
                                                                                baseline_torch_time_filepath=baseline_torch_time_filepath,
                                                                                problem_ids=dataset.problem_ids,
                                                                                p=fastp))

    # 4. used tokens
    input_tok = compute_input_tokens(run_path=run_path,
                                     max_k=max_k,
                                     num_samples=samples,
                                     problem_ids=dataset.problem_ids)
    output_tok = compute_output_tokens(run_path=run_path,
                                       max_k=max_k,
                                       num_samples=samples,
                                       problem_ids=dataset.problem_ids)
    print("=== Total tokens@k ===")
    print(f"Input tokens: {input_tok}")
    print(f"Output tokens: {output_tok}")


if __name__ == "__main__":
    main()
