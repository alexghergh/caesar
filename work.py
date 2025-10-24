import os
from dataclasses import dataclass


@dataclass
class WorkArgs:
    """Current problem the state machine iterates on."""
    problem_id: int
    sample_id: int
    problem_path: str

    def get_log_path(self) -> str:
        return os.path.join(
            "problem_" + str(self.problem_id),
            "sample_" + str(self.sample_id),
        )
