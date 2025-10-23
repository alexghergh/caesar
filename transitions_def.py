from typing import Dict

from states import CaesarState, StateOutcome, Transition


class InferenceOnlyNoGPUTransition(Transition):
    """
    Model reflection without running on the GPU. Only chat-like inference
    between the model and the user with multiple iterations.
    """
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        return {
            StateOutcome.Start: CaesarState.GENERATE_STATE,
            StateOutcome.Generate: CaesarState.START_STATE,
            StateOutcome.CPUCompileSuccess: None,
            StateOutcome.CPUCompileFail: None,
            StateOutcome.GPUCompileFail: None,
            StateOutcome.GPUCompileSuccess_CheckFail: None,
            StateOutcome.GPUCompileSuccess_CheckSuccess: None,
            StateOutcome.PerformanceSuccess: None,
            StateOutcome.PerformanceFail: None,
            StateOutcome.Finish: None,
    }


class InferenceAndGPUTransition(Transition):
    """
    Multi-turn inference. Tests GPU kernels by compiling and checking for
    corectness at every turn.
    """
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        return {
            StateOutcome.Start: CaesarState.GENERATE_STATE,
            StateOutcome.Generate: CaesarState.COMPILE_STATE,
            StateOutcome.CPUCompileSuccess: CaesarState.CORRECT_STATE,
            StateOutcome.CPUCompileFail: CaesarState.START_STATE,
            StateOutcome.GPUCompileFail: CaesarState.START_STATE,
            StateOutcome.GPUCompileSuccess_CheckFail: CaesarState.START_STATE,
            StateOutcome.GPUCompileSuccess_CheckSuccess: CaesarState.START_STATE,
            StateOutcome.PerformanceSuccess: None,
            StateOutcome.PerformanceFail: None,
            StateOutcome.Finish: None,
    }

# old stuff
# Construct Transition Cfgs
class MyTransition(Transition):
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        return {
            StateOutcome.Start: CaesarState.GENERATE_STATE,
            StateOutcome.Generate: CaesarState.COMPILE_STATE,
            # StateOutcome.Generate: CaesarState.CORRECT_STATE,
            StateOutcome.CPUCompileSuccess: CaesarState.CORRECT_STATE,
            StateOutcome.CPUCompileFail: CaesarState.START_STATE,
            StateOutcome.GPUCompileFail: CaesarState.START_STATE, # IDK
            StateOutcome.GPUCompileSuccess_CheckFail: CaesarState.START_STATE,
            StateOutcome.GPUCompileSuccess_CheckSuccess: CaesarState.PERFORMANCE_STATE,
            StateOutcome.PerformanceSuccess: CaesarState.START_STATE,
            StateOutcome.PerformanceFail: CaesarState.START_STATE,
            StateOutcome.Finish: CaesarState.FINISH_STATE, # this should never hit
    }
