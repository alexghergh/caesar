from enum import Enum


class StateOutcome(Enum):
    # These outcomes are specific to ONE state
    """Possible outcomes for a state"""

    # Outcome for SETUP_STATE
    SetupDone = "setup_done"
    SetupFinishRun = "setup_run_finished"

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

    # Outcomes for FINISH_STATE
    NextTurn = "next_turn"
    EndRun = "end_run"

    def __str__(self):
        return str(self.value)
