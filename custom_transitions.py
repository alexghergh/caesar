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
            StateOutcome.GenerateSuccess: CaesarState.FINISH_STATE,
            StateOutcome.GenerateFail: CaesarState.FINISH_STATE,
            StateOutcome.Finish: CaesarState.START_STATE,

            StateOutcome.CompileSuccess: CaesarState.NONE_STATE,
            StateOutcome.CompileFail: CaesarState.NONE_STATE,
            StateOutcome.CorrectnessSuccess: CaesarState.NONE_STATE,
            StateOutcome.CorrectnessFail: CaesarState.NONE_STATE,
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
            StateOutcome.GenerateSuccess: CaesarState.COMPILE_STATE,
            StateOutcome.GenerateFail: CaesarState.FINISH_STATE,
            StateOutcome.CompileSuccess: CaesarState.CORRECTNESS_STATE,
            StateOutcome.CompileFail: CaesarState.FINISH_STATE,
            StateOutcome.CorrectnessSuccess: CaesarState.FINISH_STATE,
            StateOutcome.CorrectnessFail: CaesarState.FINISH_STATE,
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
            StateOutcome.GenerateSuccess: CaesarState.COMPILE_STATE,
            StateOutcome.GenerateFail: CaesarState.FINISH_STATE,
            StateOutcome.CompileSuccess: CaesarState.CORRECTNESS_STATE,
            StateOutcome.CompileFail: CaesarState.FINISH_STATE,
            StateOutcome.CorrectnessSuccess: CaesarState.PERFORMANCE_STATE,
            StateOutcome.CorrectnessFail: CaesarState.FINISH_STATE,
            StateOutcome.Performance: CaesarState.FINISH_STATE,
            StateOutcome.Finish: CaesarState.START_STATE,
    }
