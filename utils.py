import os
import re
import json
import signal

from KernelBenchInternal.utils import read_file
from KernelBenchInternal.eval import KernelExecResult

from caesar_config import CaesarRunConfig
from prompts import (
    COMPILER_FEEDBACK_PROMPT,
    CORRECTNESS_FEEDBACK_PROMPT,
    EXAMPLE_CUDA_INLINE_SYNTAX,
    INITIAL_TASK_DESCRIPTION,
    INITIAL_INSTRUCTION,
    KERNEL_TO_OPTIMIZE,
    PREVIOUSLY_GENERATED_BEST_AND_LAST_KERNELS,
    PREVIOUSLY_GENERATED_KERNEL,
    PROFILER_FEEDBACK_PROMPT,
    REFLECTION_COMPILER_FEEDBACK_INSTRUCTION,
    REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION,
    REFLECTION_INSTRUCTION,
    REFLECTION_PROFILER_FEEDBACK_INSTRUCTION
)


def exec_log_to_obj(saved_dict: dict) -> KernelExecResult:
    """
    Converts a logged dict item to a KernelExecResult.
    """
    if isinstance(saved_dict, (KernelExecResult, str)):
        return saved_dict

    kernel_eval_result = KernelExecResult(
        compiled=saved_dict.get("compiled", False),
        correctness=saved_dict.get("correctness", False),
        metadata=saved_dict.get("metadata", {}),
        runtime=saved_dict.get("runtime", -1.0),
        runtime_stats=saved_dict.get("runtime_stats", {})
    )
    return kernel_eval_result


def ensure_json_serializable(obj):
    """
    Recursively convert any object into a JSON serializable format.
    Handles nested dictionaries, lists, and custom objects.

    Args:
        obj: Any Python object
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):  # Handle custom objects
        return ensure_json_serializable(obj.__dict__)
    else:
        return str(obj)  # Convert anything else to string


class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(int(self.seconds))

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


#####################
# Runs information
#####################

def load_json_data(run_path: str) -> dict:
    """Load JSON data from a run file."""
    with open(run_path) as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"Error loading run data from {run_path}: {e}")
            return None


def get_available_run_groups(base_dir: str) -> list:
    """Get list of available run groups in the base directory."""
    try:
        return sorted(
            [
                d
                for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ]
        )
    except:
        return []


def get_available_runs(base_dir: str, run_group: str) -> list:
    """Get list of available runs in the specified run group directory."""
    group_dir = os.path.join(base_dir, run_group)
    try:
        return sorted(
            [
                d
                for d in os.listdir(group_dir)
                if os.path.isdir(os.path.join(group_dir, d))
            ]
        )
    except:
        return []


def get_available_problem_ids(base_dir: str, run_group: str, run_name: str) -> list:
    """Get list of available problems in the specified run group and run name directory."""
    run_dir = os.path.join(base_dir, run_group, run_name)
    try:
        return sorted(
            [
                int(re.search(r"\d+", d).group())
                for d in os.listdir(run_dir)
                if os.path.isdir(os.path.join(run_dir, d))
            ]
        )
    except:
        return []


def get_run_group_finished_runs(base_dir: str, run_group: str) -> dict:
    """
    Get info on the number of finished runs for the given run_group.
    Returns a dictionary containing the number of evaluated samples and the
    number of total attempted samples, per run_name run.
    """
    run_group_path = os.path.join(base_dir, run_group)
    run_group_stats = {}

    # go through each run name
    for run_name in os.listdir(run_group_path):

        num_evaluated = 0
        num_total = 0
        run_path = os.path.join(run_group_path, run_name)

        for problem_id in os.listdir(run_path):
            problem_path = os.path.join(run_path, problem_id)
            for sample_id in os.listdir(problem_path):
                sample_path = os.path.join(problem_path, sample_id)
                if os.path.exists(os.path.join(sample_path, "DONE")):
                    num_evaluated += 1
                num_total += 1

        run_group_stats[run_name] = { "finished": num_evaluated, "attempted": num_total }

    return run_group_stats


def get_prev_problem_id(available_problems: list, current_problem_id: int) -> int:
    """Get the previous problem ID from the available problems list."""
    current_idx = available_problems.index(int(current_problem_id))
    return (
        available_problems[current_idx - 1]
        if current_idx > 0
        else int(current_problem_id)
    )


def get_next_problem_id(available_problems: list, current_problem_id: int) -> int:
    """Get the next problem ID from the available problems list."""
    current_idx = available_problems.index(int(current_problem_id))
    return (
        available_problems[current_idx + 1]
        if current_idx < len(available_problems) - 1
        else int(current_problem_id)
    )


def get_turn_trajectory_overviews(
    log_data: dict, max_turns: int,
) -> tuple[list, list, list]:
    """Get the trajectory of compilation, correctness, and runtime over turns."""
    turn_compile_trajectory = []
    turn_correct_trajectory = []
    turn_runtime_trajectory = []

    for turn in range(1, max_turns + 1):
        turn_data = log_data[str(turn)]

        if 'eval_result' not in turn_data or turn_data['eval_result'] == "":
            turn_compile = None
            turn_correct = None
            turn_runtime = None
        else:
            turn_compile = turn_data['eval_result'].get('compiled', None)
            turn_correct = turn_data['eval_result'].get('correctness', None)
            turn_runtime = turn_data['eval_result'].get('runtime', -1)

        turn_compile_trajectory.append(turn_compile)
        turn_correct_trajectory.append(turn_correct)
        turn_runtime_trajectory.append(turn_runtime)

    return turn_compile_trajectory, turn_correct_trajectory, turn_runtime_trajectory


def fetch_baseline_time_by_problem_id(
    baseline_time_filepath: str | os.PathLike, level: int, problem_id: int
) -> dict:
    """
    Fetch the baseline time from the timing information file.
    The problem_id parameter is the LOGICAL index of the problem in the dataset.
    This should match the problem_id in the name of the problem.
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    level_name = f"level{level}"
    try:
        for problem in baseline_json[level_name]:
            # check if the problem id matches the problem name
            if problem.split("_")[0] == str(problem_id):
                return baseline_json[level_name][problem]
    except Exception as e:
        # only reaches if the timing info is wrong
        assert False, f"Error fetching baseline time for problem {problem_id}: {e}"

    # only reaches if the timing info is absent
    assert False, f"Problem {problem_id} not found in baseline time file."


def get_turn_input_tokens(turn_data) -> int:
    toks = 0

    # anthropic
    toks += int(turn_data.get("token_usage", {}).get("input_tokens", 0))

    # sglang
    toks += int(turn_data.get("token_usage", {}).get("prompt_tokens", 0))

    return toks


def get_turn_output_tokens(turn_data) -> int:
    toks = 0

    # anthropic
    toks += int(turn_data.get("token_usage", {}).get("output_tokens", 0))

    # sglang
    toks += int(turn_data.get("token_usage", {}).get("completion_tokens", 0))

    return toks


def get_best_kernel_code(eval_result: dict) -> int | None:
    """
    Given the runtime stats of the current runs, returns the best executing
    kernel index in terms of its runtime.

    If no such kernel exists, return None.

    *Note*: The index returned assumes the ordering of the eval_result
    dictionary is the same as the kernel code dictionary.
    """
    best_runtime = float('inf')
    best_idx = None
    for eval_idx in eval_result.keys():
        eval: KernelExecResult = eval_result[eval_idx]
        if eval is not None and eval.runtime is not None:
            if eval.runtime != -1 and eval.runtime < best_runtime:
                best_runtime = eval.runtime
                best_idx = eval_idx
    return best_idx


def get_last_kernel_code(kernel_code: dict) -> int | None:
    """
    Get the index of the last kernel, regardless of compilation or runtime
    performance.

    If no such kernel exists, return None.
    """
    last_kernel_idx = None
    for idx, code in kernel_code.items():
        if code != "":
            last_kernel_idx = idx
    return last_kernel_idx
