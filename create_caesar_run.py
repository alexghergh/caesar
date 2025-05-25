import os
from pathlib import Path
from typing import Callable, List, Tuple    

"""
Systematic Script to generate Caesar runs with various configs and kick them automatically
"""

# Should be universal
NUM_WORKERS = 32
NUM_GPUS = 8
USE_SUBSET = False

# for now
GREEDY_SAMPLE = True


def exp_reflection_all_prev_config():
    """
    Reflection on all previous generations
    """
    context_strategy = ["reflection"]
    use_last_only = False
    return context_strategy, use_last_only

def exp_reflection_prev_only_config():
    """
    Reflection o previous generations
    """
    context_strategy = ["reflection"]
    use_last_only = True
    return context_strategy, use_last_only
    
def exp_eval_result_prev_only_config():
    """
    With programatic feedback
    """
    context_strategy = ["eval_result"]
    use_last_only = True
    return context_strategy, use_last_only
    

def generate_caesar_command(
        k: int,
        level: int,
        run_group: str,
        run_name_prefix: str,
        inference_provider: str,
        greedy_sample: bool,
        num_workers: int,
        num_gpus: int,
        use_subset: bool,
        feedback_strategy: Callable[[], Tuple[List[str], bool]]
    ):
    """Generate caesar command"""

    assert inference_provider in ["anthropic", "deepseek", "together"]

    context_strategy, use_last_only = feedback_strategy()
    # TODO: fix this
    context_strategy = " ".join(context_strategy)

    command_template = (
        "python3 run_multi_turn.py "
        "max_k={k} "
        "level={level} "
        "dataset_name=KernelBench/level{level} "
        'run_group="{run_group}" '
        'run_name="{run_name_prefix}_turns_{k}" '
        ".{inference_provider} "
        "greedy_sample={greedy_sample} "
        "mock=False "
        "verbose=False "
        "num_workers={num_workers} "
        "num_gpus={num_gpus} "
        "use_subset={use_subset} "
        "--list context_strategy {context_strategy} list-- "
        "use_last_only={use_last_only} "
    )
    
    return command_template.format(k=k,
                                run_group=run_group, 
                                run_name_prefix=run_name_prefix, 
                                num_workers=num_workers, 
                                num_gpus=num_gpus,
                                use_subset=use_subset,
                                level=level,
                                inference_provider=inference_provider,
                                greedy_sample=greedy_sample,
                                context_strategy=context_strategy,
                                use_last_only=use_last_only)

def save_bash_script(commands, script_path="scripts/run_caesar.sh"):
    """Save commands to a bash script and make it executable"""
    # Create scripts directory if it doesn't exist
    script_dir = Path("scripts")
    script_dir.mkdir(exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Generate Caesar runs with different max_k values\n\n")
        for cmd in commands:
            f.write(f"{cmd}\n")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Generated bash script at {script_path}")

if __name__ == "__main__":

    level = 1
    inference_provider = "deepseek"
    run_group = f"trial_level{level}_eval_result_last_only_deepseek"
    # run_group = f"level{level}_reflection_last_only_llama"
    run_name_prefix = "run_deepseek"

    command = generate_caesar_command(
        k=10,
        level=level,
        run_group=run_group,
        run_name_prefix=run_name_prefix,
        inference_provider=inference_provider,
        greedy_sample=GREEDY_SAMPLE,
        num_workers=NUM_WORKERS,
        num_gpus=NUM_GPUS,
        use_subset=USE_SUBSET,
        feedback_strategy=exp_eval_result_prev_only_config
        # feedback_strategy=exp_reflection_all_prev_config
    )

    print(command)
    
    # save_bash_script(commands) 