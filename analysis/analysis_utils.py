import json
from pathlib import Path    
import os
def load_run_data(run_path):
    """Load JSON data from a run file"""
    with open(run_path) as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"Error loading run data from {run_path}: {e}")
            return None

def get_available_runs(base_path: str):
    """Get list of available run directories"""
    base_path = Path(base_path)
    runs = []
    for run_dir in base_path.glob("*"):
        if run_dir.is_dir():
            runs.append(run_dir.name)
    return sorted(runs)



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
    level: int, problem_id: int, baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    problem_id is the LOGICAL index of the problem in the datset
    should be the problem id in the name of the problem
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
        print(f"Error fetching baseline time for problem {problem_id}: {e}")
        return None

    print(f"Problem {problem_id} not found in baseline time file")
    return None


