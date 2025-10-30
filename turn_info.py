from typing import Any


class LLMTurnInfo:
    def __init__(self) -> None:
        # each parameter contains info for each turn
        self._data = {
            "prompt": {}, # dict[int, str]
            "model_response": {}, # dict[int, str]
            "kernel_code": {}, # dict[int, str]
            "eval_result": {}, # dict[int, dict] - compile / runtime feedback
            "profiler_result": {} # dict[int, str] - profiler feedback
        }

    def __getattr__(self, name: str):
        if name in self._data:
            return self._data[name]
        else:
            # should not reach this
            raise AttributeError(f"'LLMTurnInfo' object has no attribute '{name}'.")

    def __setattr__(self, name: str, value: Any):
        if name == "_data":
            super().__setattr__(name, value)
        else:
            # should not reach this
            raise AttributeError(f"Cannot set attribute '{name}' on 'LLMTurnInfo' object.")

    def __getitem__(self, key: Any):
        # key is turn / round
        ret_val = {}
        for data in self._data:
            ret_val[data] = self._data[data].get(key, "")

        # special case default value for eval_result
        ret_val["eval_result"] = self._data["eval_result"].get(key, {})

        return ret_val

    # helper setter method
    def update_turn_data(self, turn: int, turn_data: dict[str, Any]):
        for field in self._data.keys():
            self._data[field][turn] = turn_data.get(field, "")

        # special case default value for eval_result
        self._data["eval_result"][turn] = turn_data.get("eval_result", {})
