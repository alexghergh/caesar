import json
from pathlib import Path

from utils import ensure_json_serializable, exec_log_to_obj
from conversation_info import ConversationInfo


class CaesarLogger:
    """
    Logger for _a single instance_ of a problem (i.e. one problem, one sample).
    """

    def __init__(
        self,
        log_dir: str,
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

        self.current_log: dict = {}

        self.verbose = verbose

    def save_log(self) -> None:
        """Save the current log to a JSON file."""
        with open(self.log_file, "w") as f:
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
            with open(self.log_file, "r") as f:
                self.current_log = json.load(f)
                for k in list(self.current_log.keys()):
                    # turns: str -> ints (json stores as str)
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

    def update_turn(self, turn: int, llm_info: ConversationInfo) -> None:
        if turn not in self.current_log:
            self.current_log[turn] = {}

        if 'system_prompts' not in self.current_log:
            self.current_log['system_prompts'] = {}

        self.current_log['system_prompts']['coding_agent_system_prompt'] = (
            llm_info.coding_agent_system_prompt
        )
        self.current_log['system_prompts']['prompt_agent_system_prompt'] = (
            llm_info.prompt_agent_system_prompt
        )
        self.current_log['system_prompts']['reviewer_agent_system_prompt'] = (
            llm_info.reviewer_agent_system_prompt
        )

        self.current_log[turn]['prompt_agent_input'] = llm_info.prompt_agent_input.get(turn, '')
        self.current_log[turn]['prompt_agent_output'] = llm_info.prompt_agent_output.get(turn, '')
        self.current_log[turn]['input_prompt'] = llm_info.input_prompt.get(turn, '')


        self.current_log[turn]['model_response'] = llm_info.model_response.get(turn, '')
        self.current_log[turn]['kernel_code'] = llm_info.kernel_code.get(turn, '')
        self.current_log[turn]['token_usage'] = llm_info.token_usage.get(turn, '')
        self.current_log[turn]['eval_result'] = llm_info.eval_result.get(turn, '')
        self.current_log[turn]['profiler_result'] = llm_info.profiler_result.get(turn, '')

        self.current_log[turn]['rag_query'] = llm_info.rag_query.get(turn, '')
        self.current_log[turn]['rag_context'] = llm_info.rag_context.get(turn, '')
        self.current_log[turn]['rag_scope'] = llm_info.rag_scope.get(turn, '')

        self.current_log[turn]['compile_prompt'] = llm_info.compile_prompt.get(turn, '')
        self.current_log[turn]['compile_summary'] = llm_info.compile_summary.get(turn, '')
        self.current_log[turn]['runtime_prompt'] = llm_info.runtime_prompt.get(turn, '')
        self.current_log[turn]['runtime_summary'] = llm_info.runtime_summary.get(turn, '')
        self.current_log[turn]['profiler_prompt'] = llm_info.profiler_prompt.get(turn, '')
        self.current_log[turn]['profiler_summary'] = llm_info.profiler_summary.get(turn, '')

    def update_turn_and_log(self, turn: int, llm_info) -> None:
        """
        Update the data for a specific turn, then save the log data.

        Args:
            turn: Turn number to log.
            llm_info: Contains the LLM turn information, such as prompt,
                model_response, kernel_code, eval_result, profiler_result.
        """
        self.update_turn(turn, llm_info)
        self.save_log()

