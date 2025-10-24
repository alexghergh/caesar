from enum import Enum


class CaesarState(Enum):
    """State machine states"""
    START_STATE = "start"
    GENERATE_STATE = "generate"
    COMPILE_STATE = "compile"
    CORRECTNESS_STATE = "correctness"
    PERFORMANCE_STATE = "performance"
    FINISH_STATE = "finish"
    NONE_STATE = "none" # should never reach this


class StateOutcome(Enum):
    # These outcomes are specific to ONE state
    """Possible outcomes for a state"""

    # Outcomes for START_STATE
    Start = "start"

    # Outcomes for GENERATE_STATE
    Generate = "generate"

    # Outcomes for COMPILE_STATE
    CPUCompileSuccess = "cpu_compile_success"
    CPUCompileFail = "cpu_compile_fail"

    # Outcomes for CORRECTNESS_STATE
    GPUCorrectnessSuccess = "gpu_correctness_success"
    GPUCorrectnessFail = "gpu_correctness_fail"

    # Outcomes for PERFORMANCE_STATE
    Performance = "performance"

    # FINISH_STATE
    Finish = "finish"
