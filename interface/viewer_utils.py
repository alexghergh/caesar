import json
import os
from pathlib import Path


def get_prev_problem_id(available_problems: list, current_problem_id: int):
    """Get the previous problem ID from the available problems list"""
    sorted_problems = sorted(available_problems)
    current_idx = sorted_problems.index(int(current_problem_id))
    return sorted_problems[current_idx - 1] if current_idx > 0 else int(current_problem_id)

def get_next_problem_id(available_problems: list, current_problem_id: int):
    """Get the next problem ID from the available problems list"""  
    sorted_problems = sorted(available_problems)
    current_idx = sorted_problems.index(int(current_problem_id))
    return sorted_problems[current_idx + 1] if current_idx < len(sorted_problems) - 1 else int(current_problem_id)




# def main():
#     fetch_baseline_time_by_problem_id("level1", 1, ["problem_1.py", "problem_2.py", "problem_3.py"], "baseline_time.json")\
#     # Some tests

#     PATH_TO_REPO_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

#     baseline_time_filepath = os.path.join(PATH_TO_REPO_DIR, "KernelBenchInternal", "results", "timing", "baseline_time_matx3.json")


#     print(fetch_baseline_time_by_problem_id(level=1, problem_id=9, baseline_time_filepath=baseline_time_filepath))


# if __name__ == "__main__":
#     main()
