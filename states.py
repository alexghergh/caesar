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
    GenerateSuccess = "generate_success"
    GenerateFail = "generate_fail"

    # Outcomes for COMPILE_STATE
    CompileSuccess = "compile_success"
    CompileFail = "compile_fail"

    # Outcomes for CORRECTNESS_STATE
    CorrectnessSuccess = "correctness_success"
    CorrectnessFail = "correctness_fail"

    # Outcomes for PERFORMANCE_STATE
    Performance = "performance"

    # FINISH_STATE
    Finish = "finish"
