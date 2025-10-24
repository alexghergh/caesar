from typing import Dict
from abc import ABC, abstractmethod
from states import CaesarState, StateOutcome


class Transition(ABC, Dict[StateOutcome, CaesarState]):
    """
    Abstract base class that maps (current state, current state outcome) pairs to next states.
    Subclasses must implement the transitions for all valid state-outcome combinations.
    """

    def __init__(self):
        super().__init__()
        self._validate_transitions()

    @abstractmethod
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        """
        Define the mapping of (current_state, outcome) pairs to next states.
        Must be implemented by subclasses.
        """
        pass

    def _validate_transitions(self) -> None:
        """
        Ensures all valid state-outcome combinations have a defined transition
        and loads them into the dictionary.
        """
        transitions = self._define_transitions()

        # check all the outcomes are defined
        for outcome in StateOutcome:
            if outcome not in transitions:
                raise ValueError(f"Mapping for Outcome {outcome} is not defined")

        self.update(transitions)
