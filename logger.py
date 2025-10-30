import json
from typing import Dict
from pathlib import Path

from pydra import Config

from KernelBenchInternal import eval as kernel_eval

from work import WorkArgs
from utils import ensure_json_serializable, exec_log_to_obj
from turn_info import LLMTurnInfo


class CaesarLogger:
    """
    Logger for _a single instance_ of a problem (i.e. one problem, one sample).
    """
    def __init__(
        self,
        log_dir: str,
        config: Config,
        work: WorkArgs,
        log_name: str = "log.json",
        verbose: bool = False,
    ):
        """
        Args:
            log_dir: Directory path where logs will be saved
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / log_name
        self.config_file = self.log_dir / "config.json"

        self.current_log: dict = {}

        self.verbose = verbose

        # metadata = asdict(work)
        # metadata["run_group"] = config.run_group # TODO why do I need this?
        # metadata["run_name"] = config.run_name # TODO why do I need this?
        # metadata["num_rounds"] = config.max_k # TODO why do I need this?
        # self.current_log["metadata"] = metadata

        # log the run config with initial params
        self._log_config(config.to_dict())

    def _log_config(self, config: dict) -> None:
        with open(self.config_file, 'w') as f:
            json.dump(ensure_json_serializable(config), f, indent=2)

    def save_log(self) -> None:
        """Save the current log to a JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(ensure_json_serializable(self.current_log), f, indent=2)
            if self.verbose:
                print(f"[LOG] Saved {self.log_file}")

    def load_log(self) -> None:
        """
        Load existing log data from the log file if it exists. Note that this
        doesn't do any error checking, so if a run is incomplete recovery is
        needed.
        """
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.current_log = json.load(f)
                for k in list(self.current_log.keys()):
                    # transform turns from json's str into int
                    if k.isdigit():
                        self.current_log[int(k)] = self.current_log.pop(k)

                        if "eval_result" in self.current_log[int(k)]:
                            self.current_log[int(k)]["eval_result"] = exec_log_to_obj(
                                self.current_log[int(k)]["eval_result"]
                            )

    def clean_log(self) -> None:
        """
        Clean all the existing log info, _without_ writing to the file.
        """
        self.current_log.clear()

    def update_turn(self, turn: int, llm_info: LLMTurnInfo) -> None:
        if turn not in self.current_log:
            self.current_log[turn] = {
                "prompt": "",
                "model_response": "",
                "kernel_code": "",
                "eval_result": {},
                "profiler_result": "",
            }

        if llm_info.prompt.get(turn, None):
            self.current_log[turn]["prompt"] = llm_info.prompt[turn]
        if llm_info.model_response.get(turn, None):
            self.current_log[turn]["model_response"] = llm_info.model_response[turn]
        if llm_info.kernel_code.get(turn, None):
            self.current_log[turn]["kernel_code"] = llm_info.kernel_code[turn]
        if llm_info.eval_result.get(turn, None):
            self.current_log[turn]["eval_result"] = llm_info.eval_result[turn]
        if llm_info.profiler_result.get(turn, None):
            self.current_log[turn]["profiler_result"] = llm_info.profiler_result[turn]

    def update_turn_and_log(self, turn: int, llm_info: LLMTurnInfo) -> None:
        """
        Update the data for a specific turn, then save the log data.

        Args:
            turn: Turn number to log.
            llm_info: Contains the LLM turn information, such as prompt,
                model_response, kernel_code, eval_result, profiler_result.
        """
        self.update_turn(turn, llm_info)
        self.save_log()







    def _save_timeout_eval(self, msg: str) -> None:
        """
        Special case for handling eval timeouts on GPU.
        """
        # log self.load_log()
        self.current_log[str(self.turn)]["eval_result"] = kernel_eval.KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"timeout_error": msg,
                          "hardware": "", # TODO: if necessary, add
                          "device": ""
                        }
                )

        self.save_log()

    def get_turn_data(self, turn: int) -> Dict[str, str]:
        """
        Get the logged data for a specific turn.

        Args:
            turn: Turn number to retrieve

        Returns:
            Dictionary containing the turn's logged data
        """
        return self.current_log.get(turn)
