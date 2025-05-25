####
# Upload multi-turn kernel to Hugging Face
####

import os
from huggingface_hub import HfApi
from tqdm import tqdm
api = HfApi()

from interface.run_mapping import (
    RUN_MAPPING_LEVEL_1,
    RUN_MAPPING_LEVEL_2,
    RUN_MAPPING_LEVEL_3
)

level_to_run_mapping = {
    "level1": RUN_MAPPING_LEVEL_1,
    "level2": RUN_MAPPING_LEVEL_2,
    "level3": RUN_MAPPING_LEVEL_3
}

# we upload the result of the following strategies
# that we reported in arxiv paper
LEVELS = ["level1", "level2", "level3"]
STRATEGIES = [
    "reflection_last_only", 
    "eval_result_last_only",
    "eval_result_profiler_last_only"
]
MODELS = [
    "deepseek-v3",
    "llama-3.1-70b-inst",
    "deepseek-R1"
]

PATH_TO_REPO_DIR = os.path.dirname(os.getcwd()) # run from cuda_monkeys/caesar/
BASE_LOG_DIR = "/matx/u/simonguo/kernel_multi_turn"
REPO_ID = "ScalingIntelligence/kernelbench-samples"
# grab the run_mappping



# Logic: loop through the run_mapping
# grab the folder and upload them to the remote dataset / space?

def upload_to_hf(local_path, level, strategy, model):

    # Upload all the content from the local folder to your remote Space.
    # By default, files are uploaded at the root of the repo
    api.upload_folder(
        folder_path=local_path,
        repo_id=REPO_ID,
        path_in_repo=f"iterative_refinement/{level}/{strategy}/{model}",
        # just upload log.json
        allow_patterns="*log.json",
        repo_type="dataset",
    )

def main():
    total_iterations = len(LEVELS) * len(STRATEGIES) * len(MODELS)
    progress_bar = tqdm(total=total_iterations, desc="Uploading kernels")

    for level in LEVELS:
        run_mapping = level_to_run_mapping[level]
        for strategy in STRATEGIES:
            for model in MODELS:
                print("-"*40)
                run_group = run_mapping[strategy][model]["run_group"]
                run_name = run_mapping[strategy][model]["run_name"]
                print(f"For level {level}, strategy {strategy}, model {model}, Found run_group: {run_group}, run_name: {run_name}")
                
                local_path = os.path.join(BASE_LOG_DIR, run_group, run_name)
                print(f"Uploading {local_path} to Hugging Face")
                upload_to_hf(local_path, level=level, strategy=strategy, model=model)
                
                progress_bar.update(1)
    
    progress_bar.close()

def delete_multi_turn_logs():
    """
    Delete the multi-turn logs from the HuggingFace dataset
    """
    print("Deleting multi-turn logs")
    api.delete_folder(
        repo_id=REPO_ID,
        path_in_repo="iterative_refinement",
        repo_type="dataset",
    )

if __name__ == "__main__":

    # delete_multi_turn_logs()
    main()

