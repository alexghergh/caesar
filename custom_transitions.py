from typing import Dict

from states import CaesarState, StateOutcome
from transition import Transition


class InferenceOnlyNoGPUTransition(Transition):
    """
    Model reflection without compiling or running on the GPU. Only chat-like
    inference between the model and the user with multiple iterations.
    """
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        return {
            StateOutcome.Start: CaesarState.GENERATE_STATE,
            StateOutcome.Generate: CaesarState.FINISH_STATE,
            StateOutcome.Finish: CaesarState.START_STATE,

            StateOutcome.CPUCompileSuccess: CaesarState.NONE_STATE,
            StateOutcome.CPUCompileFail: CaesarState.NONE_STATE,
            StateOutcome.GPUCorrectnessSuccess: CaesarState.NONE_STATE,
            StateOutcome.GPUCorrectnessFail: CaesarState.NONE_STATE,
            StateOutcome.Performance: CaesarState.NONE_STATE,
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
            StateOutcome.CPUCompileSuccess: CaesarState.CORRECTNESS_STATE,
            StateOutcome.CPUCompileFail: CaesarState.FINISH_STATE,
            StateOutcome.GPUCorrectnessSuccess: CaesarState.FINISH_STATE,
            StateOutcome.GPUCorrectnessFail: CaesarState.FINISH_STATE,
            StateOutcome.Finish: CaesarState.START_STATE,

            StateOutcome.Performance: CaesarState.NONE_STATE,
    }


class InferenceAndGPUAndProfiler(Transition):
    """
    Multi-turn inference. Tests GPU kernels by compiling and checking for
    corectness at every turn. If correct, profiles and sends the feedback
    to the model in the next turn.
    """
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        return {
            StateOutcome.Start: CaesarState.GENERATE_STATE,
            StateOutcome.Generate: CaesarState.COMPILE_STATE,
            StateOutcome.CPUCompileSuccess: CaesarState.CORRECTNESS_STATE,
            StateOutcome.CPUCompileFail: CaesarState.FINISH_STATE,
            StateOutcome.GPUCorrectnessSuccess: CaesarState.PERFORMANCE_STATE,
            StateOutcome.GPUCorrectnessFail: CaesarState.FINISH_STATE,
            StateOutcome.Performance: CaesarState.FINISH_STATE,
            StateOutcome.Finish: CaesarState.START_STATE,
    }
