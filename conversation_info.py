from typing import Any
from dataclasses import dataclass, field, fields


@dataclass
class ConversationInfo:
    """
    Information regarding kernel code generation for each turn.
    """

    # agents system prompts
    coding_agent_system_prompt: str = field(default_factory=str)
    reviewer_agent_system_prompt: str = field(default_factory=str)


    # per turn info
    input_prompt: dict[int, str] = field(default_factory=dict)
    model_response: dict[int, str] = field(default_factory=dict)
    kernel_code: dict[int, str] = field(default_factory=dict)
    token_usage: dict[int, dict] = field(default_factory=dict)
    eval_result: dict[int, dict] = field(default_factory=dict)
    profiler_result: dict[int, str] = field(default_factory=dict)
    compile_prompt: dict[int, str] = field(default_factory=dict)
    compile_summary: dict[int, dict] = field(default_factory=dict)
    runtime_prompt: dict[int, str] = field(default_factory=dict)
    runtime_summary: dict[int, dict] = field(default_factory=dict)
    profiler_prompt: dict[int, str] = field(default_factory=dict)
    profiler_summary: dict[int, dict] = field(default_factory=dict)

    def __getitem__(self, key: Any):
        # key is turn / round
        if not isinstance(key, int):
            raise IndexError(f"{self.__class__.__name__} doesn't expect non-integer indexing")

        # we gather the data for the specific turn from each of the fields
        ret_val: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                ret_val[f.name] = value.get(key, f.default_factory())
            else:
                ret_val[f.name] = value


        return ret_val

    # helper setter method
    def update_turn_data(self, turn: int, turn_data: dict[str, Any]):
        """
        Update turn of conversation data. Careful, this overrides the whole turn
        of data, so passing incomplete dictionaries as the turn data will
        override other fields.
        """
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, dict):
                value[turn] = turn_data.get(f.name, f.default_factory())
            else:
                setattr(self, f.name, turn_data.get(f.name, value))


