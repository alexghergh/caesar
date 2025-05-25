#!/bin/bash

# Store all commands in an array

commands=(
    "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_reflection_all_prev_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True mock=False verbose=False num_workers=6 num_gpus=4 use_subset=False --list context_strategy reflection list-- use_last_only=False model_name=deepseek-reasoner"


    # r1 with reflections
    # "python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_reflection_all_prev_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=False model_name=deepseek-reasoner"
     #"python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_reflection_all_prev_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=False model_name=deepseek-reasoner"

    #"python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_reflection_last_only_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True mock=False verbose=False num_workers=40 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=True model_name=deepseek-reasoner"
    #"python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_reflection_last_only_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True mock=False verbose=False num_workers=40 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=True model_name=deepseek-reasoner"
    #"python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_reflection_last_only_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True mock=False verbose=False num_workers=40 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=True model_name=deepseek-reasoner"
#
    # # llama reflection last only
    # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group="level2_reflection_last_only_llama" run_name="run_llama_turns_10" .together greedy_sample=True mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=True"
    # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group="level3_reflection_last_only_llama" run_name="run_llama_turns_10" .together greedy_sample=True mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=True"

    # # llama reflection all prev
    # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group="level2_reflection_all_prev_llama" run_name="run_llama_turns_10" .together greedy_sample=True mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=False"
    # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group="level3_reflection_all_prev_llama" run_name="run_llama_turns_10" .together greedy_sample=True mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy reflection list-- use_last_only=False"

    # # r1 with profiler! 
    # "python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_eval_result_profiler_last_only_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=48 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True model_name=deepseek-reasoner"
    # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_eval_result_profiler_last_only_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=48 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True model_name=deepseek-reasoner"
    # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_eval_result_profiler_last_only_deepseek run_name=run_v0_deepseek_r1_turn .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=48 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True model_name=deepseek-reasoner"
    
    # # Prioritize
    # # # "python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_eval_result_last_only_llama run_name=run_v0_llama_turns_10 .together greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result list-- use_last_only=True"
    # # # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_eval_result_last_only_llama run_name=run_v0_llama_turns_10 .together greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result list-- use_last_only=True"
    # # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_eval_result_last_only_llama run_name=run_v0_llama_turns_10 .together greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result list-- use_last_only=True"


    # # "python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_eval_result_last_only_deepseek run_name=run_v0_deepseek_r1_turn_10 .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result list-- use_last_only=True model_name=deepseek-reasoner"
    # # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_eval_result_last_only_deepseek run_name=run_v0_deepseek_r1_turn_10 .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result list-- use_last_only=True model_name=deepseek-reasoner"
    # # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_eval_result_last_only_deepseek run_name=run_v0_deepseek_r1_turn_10 .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result list-- use_last_only=True model_name=deepseek-reasoner"


    # # # "python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_eval_result_profiler_last_only_llama run_name=run_v0_llama_turns_10 .together greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True"
    # # # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_eval_result_profiler_last_only_llama run_name=run_v0_llama_turns_10 .together greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True"
    # # # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_eval_result_profiler_last_only_llama run_name=run_v0_llama_turns_10 .together greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True"


    # # # "python3 run_multi_turn.py max_k=10 level=1 dataset_name=KernelBench/level1 run_group=level1_eval_result_profiler_last_only_deepseek run_name=run_v0_deepseek_turn_10 .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True"
    # # # "python3 run_multi_turn.py max_k=10 level=2 dataset_name=KernelBench/level2 run_group=level2_eval_result_profiler_last_only_deepseek run_name=run_v0_deepseek_turn_10 .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True"
    # # # "python3 run_multi_turn.py max_k=10 level=3 dataset_name=KernelBench/level3 run_group=level3_eval_result_profiler_last_only_deepseek run_name=run_v0_deepseek_turn_10 .deepseek greedy_sample=True num_samples=1 mock=False verbose=False num_workers=32 num_gpus=8 use_subset=False --list context_strategy eval_result profiler list-- use_last_only=True"


)


# Function to kill all python processes
kill_processes() {
    pkill -f "python3 run_multi_turn.py"
}

# Set up trap to kill processes on script exit
trap kill_processes EXIT

# Remove while true and shuffling, just iterate through commands sequentially
for cmd in "${commands[@]}"; do
    echo "Starting: $cmd"
    
    # Start time for this command
    start_time=$SECONDS
    
    # Run command in background
    eval "$cmd" &
    cmd_pid=$!
    
    # Wait for up to 10 minutes
    while kill -0 $cmd_pid 2>/dev/null; do
        elapsed=$(( SECONDS - start_time ))
        if (( elapsed >= 600 )); then
            echo "10 minutes elapsed, killing process..."
            pkill -f "python3 run_multi_turn.py"
            kill -9 $cmd_pid 2>/dev/null
            # pkill -f python
            # pkill -f python3
            sleep 5
            break
        fi
        sleep 1
    done
    
    echo "Moving to next command..."
    sleep 5  # Brief pause before next command
done

echo "All commands completed."

