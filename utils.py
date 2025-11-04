import os
import re
import json
import signal
from typing import List

from KernelBenchInternal.utils import read_file
from KernelBenchInternal.eval import KernelExecResult

from prompts import (
    ALL_PREVIOUSLY_GENERATED_KERNELS_HEADER,
    ALL_PREVIOUSLY_GENERATED_KERNELS_ITERATION,
    COMPILER_FEEDBACK_BEST_ONLY_PROMPT,
    COMPILER_FEEDBACK_ITERATION_PROMPT,
    CORRECTNESS_FEEDBACK_BEST_ONLY_PROMPT,
    CORRECTNESS_FEEDBACK_ITERATION_PROMPT,
    EXAMPLE_CUDA_INLINE_SYNTAX,
    INITIAL_TASK_DESCRIPTION,
    INITIAL_INSTRUCTION,
    KERNEL_TO_OPTIMIZE,
    PREVIOUSLY_GENERATED_BEST_KERNEL,
    PROFILER_FEEDBACK_BEST_ONLY_PROMPT,
    PROFILER_FEEDBACK_ITERATION_PROMPT,
    REFLECTION_COMPILER_FEEDBACK_INSTRUCTION,
    REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION,
    REFLECTION_INSTRUCTION,
    REFLECTION_PROFILER_FEEDBACK_INSTRUCTION
)
from strategy import Strategy


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


###########################
# Prompt construction
###########################

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench", "KernelBench")
KERNEL_BENCH_ARCH_EXAMPLES_PATH = os.path.join(
    REPO_TOP_PATH, "KernelBench", "KernelBenchInternal", "prompts"
)


def generate_initial_prompt(
        ref_arch_src: str, strategy: set[Strategy]
) -> str:
    """
    Construct an initial template prompt to show to the model.
    Additionally, it contains an example implementation of a custom CUDA kernel
    in PyTorch.
    """

    # example kernel to show syntax (addition kernel)
    example_ind = 'add'
    example_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_new_ex_{example_ind}.py"
    )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    # construct the initial prompt
    prompt = INITIAL_TASK_DESCRIPTION

    if Strategy.SHOW_INLINE_SYNTAX in strategy:
        prompt += EXAMPLE_CUDA_INLINE_SYNTAX.format(
            example_arch_src=example_arch, example_new_arch_src=example_new_arch
        )

    prompt += KERNEL_TO_OPTIMIZE.format(arch_src=ref_arch_src)

    return prompt


def build_llm_prompt_for_turn(
    turn: int,
    ref_arch_src: str,
    kernels: dict,
    eval_result: dict,
    profiler_result: dict,
    strategy: set[Strategy],
    max_profiler_feedback_length: int,
) -> str:
    """
    Build the current context for the given turn / round. This takes into
    account all the information from the previous steps.
    """
    # initial prompt (always start from the task description + reference kernel
    # to optimize)
    prompt = generate_initial_prompt(ref_arch_src, strategy)

    if turn == 1:
        prompt += INITIAL_INSTRUCTION
        return prompt

    else:

        # check whether we have any kernels generated so far
        if kernels is None or all(not v for v in kernels.values()):
            # don't have a valid kernel code so far, so re-prompt using the
            # initial prompt
            prompt += INITIAL_INSTRUCTION
            return prompt

        else:

            # best kernel so far in terms of runtime (the above `if` branch
            # checks whether we do have a kernel; this means we have kernel
            # code; however, it doesn't mean it compiled or ran correctly!
            # that's why this can return None)
            best_kernel_idx: int | None = get_best_kernel_code(eval_result)

            # if we don't have a _best_ kernel, we might still have some kernel
            # code to give the LLM feedback on (e.g. compile errors, runtime
            # errors etc.)
            kernel_idx = best_kernel_idx
            if best_kernel_idx is None:
                for idx, kern in reversed(list(kernels.items())):
                    if kern != "":
                        kernel_idx = idx
                        break

            # TODO make sure that kernel_idx is guaranteed at this point to be a
            # valid index to kernel code

            # NOTE: running with best_only will likely generate the same LLM
            # prompt for multiple turns, if the LLM fails to improve the code;
            # with temperature 0, the state machine will likely get stuck
            # forever with the same LLM output!!!
            if Strategy.BEST_ONLY in strategy:
                prompt += PREVIOUSLY_GENERATED_BEST_KERNEL.format(
                    prev_kernel_code=kernels[kernel_idx]
                )
            else:
                # use all previous kernel generations
                # ignore turns where kernels are empty
                prompt += ALL_PREVIOUSLY_GENERATED_KERNELS_HEADER
                for idx, kern in kernels:
                    if kern != "":
                        prompt += ALL_PREVIOUSLY_GENERATED_KERNELS_ITERATION.format(
                            iteration=idx, kernel=kern
                        )

            # feedback is ALWAYS for the best kernel (whether it is the last
            # generated kernel or not); in the special case that multiple
            # iterations are in the prompt, but the best kernel is not the last,
            # we specifically mark the feedback as being for the best kernel
            # (i.e. we specify the iteration index of the best kernel)

            # offer compiler feedback if compilation failed; otherwise, move
            # on to correctness check
            if (
                Strategy.COMPILER_FEEDBACK in strategy
                and eval_result[kernel_idx].metadata != {} # check whether we actually compiled
                                                        # i.e. the state machine might've
                                                        # never transitioned into compilation
                and eval_result[kernel_idx].compiled is False
            ):
                metadata = eval_result[kernel_idx].metadata
                metadata.pop("hardware", None)
                metadata.pop("device", None)
                key = next(iter(metadata))
                # if it's _not_ a best_only strategy, we need to mention for
                # which kernel this feedback is
                if Strategy.BEST_ONLY not in strategy:
                    prompt += COMPILER_FEEDBACK_ITERATION_PROMPT.format(
                        iteration=turn,
                        compiler_feedback=f"{key}: {metadata[key]}"
                    )
                else:
                    prompt += COMPILER_FEEDBACK_BEST_ONLY_PROMPT.format(
                        compiler_feedback=f"{key}: {metadata[key]}"
                    )
                prompt += REFLECTION_COMPILER_FEEDBACK_INSTRUCTION
                return prompt

            # this assumes that correctness check is empty if compile failed
            # offer correctness check feedback if it failed; otherwise, move
            # on to profiler feedback
            if (
                Strategy.CORRECTNESS_FEEDBACK in strategy
                and eval_result[kernel_idx].metadata != {} # check whether we actually compiled
                                                        # i.e. the state machine might've
                                                        # never transitioned into compilation
                and eval_result[kernel_idx].compiled is True
                and eval_result[kernel_idx].correctness is False
            ):
                metadata = eval_result[kernel_idx].metadata
                metadata.pop("hardware", None)
                metadata.pop("device", None)
                key = next(iter(metadata))
                # if it's _not_ a best_only strategy, we need to mention for
                # which kernel this feedback is
                if Strategy.BEST_ONLY not in strategy:
                    prompt += CORRECTNESS_FEEDBACK_ITERATION_PROMPT.format(
                        iteration=turn,
                        correctness_feedback=f"{key}: {metadata[key]}"
                    )
                else:
                    prompt += CORRECTNESS_FEEDBACK_BEST_ONLY_PROMPT.format(
                        correctness_feedback=f"{key}: {metadata[key]}"
                    )
                prompt += REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION
                return prompt

            # this assumes that profiler data is empty if correctness check
            # failed
            if (
                Strategy.PROFILER_FEEDBACK in strategy
                and profiler_result.get(kernel_idx, "") != ""
            ):
                # TODO should we restrict the profiler feedback output if it
                # gets too long? select parts of it? use the
                # max_profiler_feedback_length info here

                # if it's _not_ a best_only strategy, we need to mention for
                # which kernel this feedback is
                if Strategy.BEST_ONLY not in strategy:
                    prompt += PROFILER_FEEDBACK_ITERATION_PROMPT.format(
                        iteration=turn,
                        profiler_feedback=profiler_result[kernel_idx][:max_profiler_feedback_length],
                    )
                else:
                    prompt += PROFILER_FEEDBACK_BEST_ONLY_PROMPT.format(
                        profiler_feedback=profiler_result[kernel_idx][:max_profiler_feedback_length],
                    )
                prompt += REFLECTION_PROFILER_FEEDBACK_INSTRUCTION
                return prompt

            # if no feedback is given, simply re-prompt the model as normal
            prompt += REFLECTION_INSTRUCTION
            return prompt


def get_best_kernel_code(eval_result: dict) -> int | None:
    """
    Given the runtime stats of the current runs, returns the best executing
    kernel index in terms of its runtime.

    If no such kernel exists, returns None.

    *Note*: The index returned assumes the ordering of the eval_result
    dictionary is the same as the kernel code dictionary.
    """
    best_runtime = 1000000000
    best_idx = None
    for eval_idx in eval_result.keys():
        eval: KernelExecResult = eval_result[eval_idx]
        if eval is not None and eval.runtime is not None:
            if eval.runtime != -1 and eval.runtime < best_runtime:
                best_runtime = eval.runtime
                best_idx = eval_idx
    return best_idx
