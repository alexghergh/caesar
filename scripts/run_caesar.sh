#!/bin/bash

# Generate Caesar runs with different max_k values

python3 test_state_machine.py max_k=1 run_name="trial_run_turns_1" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=2 run_name="trial_run_turns_2" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=3 run_name="trial_run_turns_3" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=4 run_name="trial_run_turns_4" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=5 run_name="trial_run_turns_5" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=6 run_name="trial_run_turns_6" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=7 run_name="trial_run_turns_7" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=8 run_name="trial_run_turns_8" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=9 run_name="trial_run_turns_9" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
python3 test_state_machine.py max_k=10 run_name="trial_run_turns_10" .deepseek greedy_sample=True mock=False verbose=False num_workers=24 num_gpus=8 show_state=False use_subset=False 
