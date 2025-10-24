from typing import Any


class LLMTurnInfo:
    def __init__(self) -> None:
        # each parameter contains info for each turn
        self._data = {
            "prompt": {}, # dict[int, str]
            "model_response": {}, # dict[int, str]
            "kernel_code": {}, # dict[int, str]
            "feedback": {}, # dict[int, str]
            "eval_result": {}, # dict[int, dict] - external feedback (e.g. compile, prof)
            "profiler_result": {} # dict[int, str]
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

# # TODO remove if ok
# if __name__=="__main__":
#     lol = LLMTurnInfo()
#     # lol.prompt = "lol" # OK
#     lol.prompt[1] = "lol" # OK
#     print(lol.prompt)
#     print('turn 1', lol[1])
#     print('turn 2', lol[2])
#     print('turn -1', lol[-1])
#
#     print('pyes', lol.prompt[1])
#     print('pyes', lol[1]['prompt'])
#
#     lol.update_turn_data(1, {
#         "prompt": "huh",
#         "eval_result": { "ney" },
#     })
#
#     print(lol[2])
#
#     # print(",,,", lol.prompt)
#     # lol.prompt[1] = 'lol'
#     #
#     # print(lol[1])
#     #
#     # lol[1]['prompt'] = 'haha'
#     #
#     # print(lol.prompt)
