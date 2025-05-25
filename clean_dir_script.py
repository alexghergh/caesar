import os

import shutil

"""
Using this script to clean up the directory.
especially if they are not completed.

USE WITH CAUTION.
"""
from utils import check_result_exists
LOG_DIR_PREFIX = "/matx/u/simonguo/kernel_multi_turn"
paths= os.path.join(LOG_DIR_PREFIX, "level3_reflection_all_prev_deepseek", "run_v0_deepseek_r1_turn")

run_group = "level3_reflection_all_prev_deepseek"
run_name = "run_v0_deepseek_r1_turn"

for problem_id in range(1, 51):

    if check_result_exists(LOG_DIR_PREFIX, run_group, run_name, problem_id, 0):
        print(f"exists for problem {problem_id}")
    else:
        problem_dir = os.path.join(LOG_DIR_PREFIX, run_group, run_name, f"problem_{problem_id}")
        if os.path.exists(problem_dir):
            shutil.rmtree(problem_dir)
        print(f"does not exist for problem {problem_id}")
