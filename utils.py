import os
import re
import json
import signal
from typing import List

from KernelBenchInternal.utils import read_file
from KernelBenchInternal.eval import KernelExecResult


def exec_log_to_obj(saved_dict: dict) -> KernelExecResult:
    """
    Converts a logged dict item to a KernelExecResult.
    """
    if isinstance(saved_dict, (KernelExecResult, str)):
        return saved_dict

    kernel_eval_result = KernelExecResult(
        compiled=saved_dict.get("compiled", None),
        correctness=saved_dict.get("correctness", None),
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







def get_turns(log_data: dict):
    """Get all turn numbers, excluding 'metadata'"""
    turns = [k for k in log_data.keys() if k.isdigit()]
    return sorted(turns, key=int)


def get_turn_trajectory_overviews(log_data: dict, max_turns: int = None):
    """Get the trajectory of compilation, correctness, and runtime over turns"""
    turn_compile_trajectory = []
    turn_correct_trajectory = []
    turn_runtime_trajectory = []

    # Get all turn numbers, excluding 'metadata'
    # turns = [k for k in log_data.keys() if k.isdigit()]

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

        # TODO: maybe put a try catch here?
        turn_compile_trajectory.append(turn_compile)
        turn_correct_trajectory.append(turn_correct)
        turn_runtime_trajectory.append(turn_runtime)

    return turn_compile_trajectory, turn_correct_trajectory, turn_runtime_trajectory





def fetch_baseline_time_by_problem_id(
    baseline_time_filepath: str, level: int, problem_id: int
) -> dict:
    """
    Fetch the baseline time from the timing information file.
    Given problem_id is the LOGICAL index of the problem in the dataset.
    This should match the problem id in the name of the problem.
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
# Prompt Construction




REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench", "KernelBench")
KERNEL_BENCH_ARCH_EXAMPLES_PATH = os.path.join(REPO_TOP_PATH, "KernelBench", "KernelBenchInternal", "prompts")


# These are from KernelBenchInternal/src/prompt_constructor.py
# overall problem state
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

PROBLEM_INSTRUCTION = """Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

REFLECTION_INSTRUCTION = """Given your previous generation(s), improve and optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

REFLECTION_INSTRUCTION_LAST_ONLY = """Given your latest generation, improve and optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""



def prompt_generate_initial_from_template(ref_arch_src: str):
    example_ind = 1
    example_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_new_ex_{example_ind}.py"
    )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_initial(ref_arch_src, example_arch, example_new_arch)

def prompt_generate_initial(arch_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this:
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arch_src}
    ```
    """
    # prompt += PROBLEM_INSTRUCTION
    return prompt




def build_context_multi_turn(
        initial_prompt: str,
        prompts: dict,
        kernels: dict,
        compiler_feedback: dict,
        eval_result: dict,
        profiler_result: dict,
        iteration: int,
        strategy: List[str] = [],
        use_last_only: bool = False,
        max_feedback_length: int = 4000,
    ) -> str:
    """
    Build the current context for the given iteration and strategy
    """
    # TODO rebuild this function
    prompt = initial_prompt

    # Add feedback
    if "eval_result" in strategy:
        if iteration == 1:
            prompt += "\n\n"
            prompt += PROBLEM_INSTRUCTION
        else:
            prompt += f"Here is your latest generation:\n"
            prompt += f"```\n{kernels[iteration - 1]}\n```\n"
            prompt += construct_programatic_prompt_feedback(compiler_feedback=compiler_feedback[iteration - 1],
                                                            exec_result=eval_result[iteration - 1],
                                                            profiler_feedback = profiler_result[iteration - 1] if iteration - 1 in profiler_result else "",
                                                            max_feedback_length=max_feedback_length,
                                                            use_pytorch_profiler=("profiler" in strategy)
                                                            )
        return prompt

    # Reflection
    # Iterations are 1-indexed.
    for i in range(1, iteration, 1):

        # Add what you generated
        if "reflection" in strategy:

            # Only use the latest generation
            if use_last_only:
                if i < iteration - 1: continue
                else:
                    prompt += f"Here is your latest generation:\n"
                    prompt += f"```\n{kernels[i]}\n```\n"
            else:
                # Include all previous generations

                # Special case for the first generation
                if i == 1:
                    prompt += f"Here are your previous generations:\n"

                # add kernels
                if i in kernels:
                    prompt += f"Generation {i}:\n```\n{kernels[i]}\n```\n\n"

        # Human feedback
        if "human_feedback" in strategy:
            print(f"\nPlease provide feedback for kernel {i}:")
            user_feedback = input().strip()
            if user_feedback:
                prompt += f"This is expert human feedback on your previously generated kernel #{i}:\n"
                prompt += f"{user_feedback}\n\n"

    # Instruction prompt
    if iteration == 1: # for intial
        prompt += PROBLEM_INSTRUCTION
    else:
        if "reflection" in strategy:
            if use_last_only:
                prompt += REFLECTION_INSTRUCTION_LAST_ONLY
            else:
                prompt += REFLECTION_INSTRUCTION
        else:
            raise NotImplementedError("Anything else is not implemented yet")
            prompt += PROBLEM_INSTRUCTION

    return prompt

EVAL_RESULT_INSTRUCTION_LAST_ONLY = """Name your new improved output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

def construct_programatic_prompt_feedback(compiler_feedback: str,
                                          exec_result: KernelExecResult,
                                          profiler_feedback: str = "",
                                          max_feedback_length: int = 2000,
                                          use_pytorch_profiler: bool = False) -> str:
    """
    Construct programatic feedback from the given exec_result
    input:
    - compiler_feedback: str, feedback from the compiler
    - exec_result: KernelExecResult, the result of the execution
    - profiler_feedback: str, table breakdown from the PyTorch profiler; only used if use_pytorch_profiler is True
    - max_log_length: int, the maximum length of the log to put back into context
    - use_pytorch_profiler: bool, whether to use the PyTorch profiler
    """
    feedback = "Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model. \n\n"
    feedback += "Here is the Evaluation Result: \n"

    # we do not want hardware information to influence this
    metadata_without_hw_info = exec_result.metadata
    metadata_without_hw_info.pop('hardware', None)
    metadata_without_hw_info.pop('device', None)

    # Case 1: Compilation Failure
    if not exec_result.compiled:
        feedback += "Your kernel failed to compile.\n\n"
        # TODO: restrict length of compiler feedback
        feedback += f"Here is the compiler logs\n\n:"
        feedback += f"{compiler_feedback[:max_feedback_length]}\n\n"
        for key, value in metadata_without_hw_info.items():
            feedback += f"{key}: {value}\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please fix the errors and try again.   "
        return feedback

    # Special Case: CUDA Error
    if "cuda_error" in metadata_without_hw_info:
        feedback += "Your kernel failed to run. \n\n"
        feedback += f"Here is the CUDA error: {exec_result.metadata['cuda_error']} \n\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY

        feedback += "Please fix the errors and try again."
        return feedback

    # Special Case: Time-out
    if "timeout_error" in metadata_without_hw_info:
        feedback += "Your kernel execution timed out. \n\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please fix the errors and try again."

        return feedback

    # Case 2: Compiled, But not Correct
    if not exec_result.correctness:
        feedback += "Your kernel failed to produce the correct output, compared to the reference architecture.\n\n"
        feedback += f"Here is the correctness feedback: \n\n"
        # add metadata objects
        for key, value in metadata_without_hw_info.items():
            feedback += f"{key}: {value}\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please try generating ModelNew again, while fixing the correctness issues."
        return feedback

    # Case 3: Runtime Success
    if exec_result.correctness:
        feedback += "Your kernel executed successfully and produced the correct output. \n\n"
        feedback += f"Here is your wall clock time: {exec_result.runtime} milliseconds \n\n"

        # ADD PROFILER
        if use_pytorch_profiler:
            assert len(profiler_feedback) > 0, "Profiler feedback is empty"
            feedback += f"Your Kernel was profiled with the PyTorch Profiler over many iterations, below is a table breakdown of the CUDA kernel execution time: \n\n"

            feedback += f"```\n{profiler_feedback[:max_feedback_length]}\n```\n\n"

        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please rewrite the entire kernel to be as fast as possible. \n\n"
        return feedback

    raise ValueError("[Programatic Feedback] You should not reach here")
    return None


class timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(int(self.seconds))

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
