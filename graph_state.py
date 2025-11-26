import os
from typing_extensions import TypedDict

from caesar_config import CaesarRunConfig
from conversation_info import ConversationInfo
from logger import CaesarLogger
from orchestrator import GPUOrchestrator
from states import StateOutcome
from work import WorkArgs


# langgraph state
class CaesarGraphState(TypedDict):
    # state that shouldn't change after the initial setup
    process_id: int
    config: CaesarRunConfig
    work: WorkArgs
    logger: CaesarLogger
    build_dir: str | os.PathLike
    orchestrator: GPUOrchestrator

    # state that is updated when iterating between the state machine rounds
    conversation_info: ConversationInfo
    current_turn: int
    ref_problem_src: str
    state_outcome: StateOutcome
