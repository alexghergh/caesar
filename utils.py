import os
import re
import json
import signal

from KernelBenchInternal.utils import read_file
from KernelBenchInternal.eval import KernelExecResult

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
            # get the best kernel so far in terms of runtime (the above `if`
            # branch checks whether we do have a kernel; this means we have
            # kernel code; however, it doesn't mean it compiled or ran
            # correctly! that's why this can return None) and the last generated
            # kernel (regardless of whether it compiled or ran correctly or not)
            best_kernel_idx: int | None = _get_best_kernel_code(eval_result)
            last_kernel_idx: int | None = _get_last_kernel_code(kernels)

            # at this point, last_kernel_idx is guaranteed non-None
            # there's a few cases to consider:
            # - best_kernel_idx is None (because no kernel compiled so far)
            # - best_kernel_idx is the same as last_kernel_idx (because the last
            # kernel compiled and ran correctly)
            # - best_kernel_idx is different from last_kernel_idx (because we
            # have a kernel that compiled and ran correctly at some previous
            # iteration, but the last generated one either didn't compile,
            # didn't run successfully, or was slower)

            if best_kernel_idx is None or best_kernel_idx == last_kernel_idx:
                # we don't have a best kernel yet OR it is the same as the
                # last kernel
                prompt += PREVIOUSLY_GENERATED_KERNEL.format(
                    prev_kernel_code=kernels[last_kernel_idx]
                )
            elif best_kernel_idx is not None and best_kernel_idx != last_kernel_idx:
                # different kernels generated at different times
                prompt += PREVIOUSLY_GENERATED_BEST_AND_LAST_KERNELS.format(
                    best_kernel_code=kernels[best_kernel_idx],
                    last_kernel_code=kernels[last_kernel_idx],
                )

            # we can either give the LLM feedback for the best kernel, or for
            # all the kernels generated; as the prompts can get quite large
            # (thousands of tokens) with multiple generated kernels, we should
            # only offer the best feedback available at hand, which is either:
            # - (best case) feedback for the kernel that compiled, ran, and has
            # profiler output
            # - (worst case) feedback for the last kernel that didn't compile or
            # didn't run correctly
            # - (in-between) if there's a valid (i.e. compiler + runtime
            # correct) kernel at some previous iteration, but the kernel at the
            # current iteration didn't compile or run correctly, or was slower
            # than the best kernel, than tell the model about both
            #
            # as an action plan, we always offer compiler, correctness and
            # profiler feedback for the last kernel, and profiler feedback for
            # the best kernel when the last kernel is slower

            # offer compiler feedback if compilation failed; otherwise, move
            # on to correctness check
            if (
                Strategy.COMPILER_FEEDBACK in strategy
                and eval_result[last_kernel_idx].metadata != {} # always True
                and eval_result[last_kernel_idx].compiled is False # always True
            ):
                metadata = eval_result[last_kernel_idx].metadata
                metadata.pop("hardware", None)
                metadata.pop("device", None)
                key = next(iter(metadata))

                prompt += COMPILER_FEEDBACK_PROMPT.format(
                    compiler_feedback=f"{key}: {metadata[key]}"
                )
                prompt += REFLECTION_COMPILER_FEEDBACK_INSTRUCTION
                return prompt

            # this assumes that correctness check is empty if compile failed
            # offer correctness check feedback if it failed; otherwise, move
            # on to profiler feedback
            if (
                Strategy.CORRECTNESS_FEEDBACK in strategy
                and eval_result[last_kernel_idx].metadata != {}
                and eval_result[last_kernel_idx].compiled is True
                and eval_result[last_kernel_idx].correctness is False
            ):
                metadata = eval_result[last_kernel_idx].metadata
                metadata.pop("hardware", None)
                metadata.pop("device", None)
                issue = metadata.get("correctness_issue", "")
                issue = metadata.get("runtime_error", "") if issue == "" else issue

                prompt += CORRECTNESS_FEEDBACK_PROMPT.format(
                    correctness_feedback=f"{issue}"
                )
                prompt += REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION
                return prompt

            # best is none, last runtime issue
            # best != last
            # best == last

            if Strategy.PROFILER_FEEDBACK in strategy:
                # always include best kernel profiler feedback if available
                if (
                    best_kernel_idx is not None
                    and profiler_result.get(best_kernel_idx, "") != ""
                ):
                    prompt += PROFILER_FEEDBACK_PROMPT.format(
                        kernel="best",
                        profiler_feedback=profiler_result[best_kernel_idx][
                            :max_profiler_feedback_length
                        ],
                        runtime_ms=eval_result[best_kernel_idx].runtime,
                    )

                # include last kernel profiler feedback if it was slower;
                # if it was faster, then by definition the last kernel IS the
                # best kernel
                if (
                    last_kernel_idx != best_kernel_idx

                    # if there's no profiler feedback, we can be sure something
                    # was wrong during compilation or runtime; skip the rest of
                    # the checks
                    and profiler_result.get(last_kernel_idx, "") != ""

                    # last kernel is slower than the best kernel
                    and eval_result[last_kernel_idx].runtime >
                        eval_result[best_kernel_idx].runtime
                ):
                    prompt += PROFILER_FEEDBACK_PROMPT.format(
                        kernel="previous",
                        profiler_feedback=profiler_result[last_kernel_idx][
                            :max_profiler_feedback_length
                        ],
                        runtime_ms=eval_result[last_kernel_idx].runtime,
                    )

                prompt += REFLECTION_PROFILER_FEEDBACK_INSTRUCTION
                return prompt

            # if no feedback is given, simply re-prompt the model as normal
            prompt += REFLECTION_INSTRUCTION
            return prompt


def _get_best_kernel_code(eval_result: dict) -> int | None:
    """
    Given the runtime stats of the current runs, returns the best executing
    kernel index in terms of its runtime.

    If no such kernel exists, return None.

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


def _get_last_kernel_code(kernel_code: dict) -> int | None:
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
