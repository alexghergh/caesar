import os
from typing import List
import json
from pydra import Config
import signal



from utils import construct_programatic_feedback
from KernelBenchInternal.src.eval import KernelExecResult
###########################
# Prompt Construction


# Template 
    # sample_exec_result = KernelExecResult(
    #     compiled=False,
    #     correctness=False,
    #     metadata="", # need work on this
    #     runtime=-1.0,
    #     runtime_stats={}
    # )




###########################
# Test Cases
# A representative set of samples
###########################
# 

def test_compiled_fail_feedback():

    sample_exec_result = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:0', 
            'compilation_error': RuntimeError("Error building extension 'max_reduction'")
        },
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)



def test_correctness_val_fail_feedback():
    
    sample_exec_result = KernelExecResult(
        compiled=True,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S', 
            'device': 'cuda:0', 
            'max_difference': ['29.875006', '28.969961', '27.741943', '27.972809', '27.862772'], 
            'avg_difference': ['2.578733', '2.575977', '2.574863', '2.581244', '2.582202'], 
            'correctness_issue': 'Output mismatch', 
            'correctness_trials': '(0 / 5)'
        },
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)


def test_correctness_shape_fail_feedback():

    sample_exec_result = KernelExecResult(
        compiled=True,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:2',
            'correctness_issue': 'Output shape mismatch: Expected torch.Size([128, 64, 32, 32]), got torch.Size([128, 64, 64, 64])'
        },
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)


def test_cuda_error_feedback():

    sample_exec_result = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:0',
            'cuda_error': 'CUDA Error: CUDA error: an illegal memory access was encountered\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\nFor debugging consider passing CUDA_LAUNCH_BLOCKING=1\nCompile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n'
        },
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)


def test_function_name_exists_feedback():

    sample_exec_result = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:0',
            'other_error': 'error: "attribute \'bias\' already exists"'
        },
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)

def test_runtime_arg_fail_feedback():

    sample_exec_result = KernelExecResult(
        compiled=True,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:0',
            'runtime_error': 'fused_conv_gelu_norm_cuda(): incompatible function arguments. The following argument types are supported:\n    1. (arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: int, arg4: int, a...'},
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)


def test_empty_feedback():
    """
     Log Path: /matx/u/simonguo/kernel_multi_turn/level2_reflection_all_prev_deepseek/run_deepseek_turns_3/problem_24/sample_0/log.json 
    """

    sample_exec_result = KernelExecResult(
        compiled=True,
        correctness=False,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:4',
            'runtime_error': {}
        },
        runtime=-1.0,
        runtime_stats={}
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)


def test_correctness_success_feedback():
    sample_exec_result = KernelExecResult(
        compiled=True,
        correctness=True,
        metadata={
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:3',
            'correctness_trials': '(5 / 5)'
        },
        runtime=2.45,
        runtime_stats={
            'mean': 2.45,
            'std': 0.00197,
            'min': 2.45,
            'max': 2.46,
            'num_trials': 100,
            'hardware': 'NVIDIA L40S',
            'device': 'cuda:3'
        }
    )

    feedback = construct_programatic_feedback(sample_exec_result)
    print(feedback)

