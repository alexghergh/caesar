import time
import os
import copy

import torch

from KernelBenchInternal import eval as kernel_eval
from KernelBenchInternal.utils import (
    extract_last_code,
    query_server,
    read_file,
)

from eval import (
    compile_single_sample,
    evaluate_single_sample_src,
    get_torch_profiler_info
)
from states import CaesarState, StateOutcome
from work import WorkArgs
from logger import CaesarLogger
from utils import build_llm_prompt_for_turn
from orchestrator import GPUOrchestrator
from transition import Transition
from caesar_config import CaesarRunConfig
from turn_info import LLMTurnInfo


class CaesarStateMachine:
    def __init__(
        self,
        transitions: Transition,
        config: CaesarRunConfig,
        work: WorkArgs,
        logger: CaesarLogger,
        process_id: int,
        orchestrator: GPUOrchestrator,
    ):
        self.transitions = transitions
        self.config = config
        self.work = work

        self.state = CaesarState.START_STATE

        self.current_k = 1
        self.max_k = config.max_k

        # contains the reference problem in Python code as a string
        # load it from KernelBench repo
        self.ref_problem_src = read_file(self.work.problem_path)

        # build dir to cache compiled problems
        self.build_dir = os.path.join(
            self.config.build_dir_prefix,
            self.config.run_group,
            self.config.run_name,
            self.work.get_log_path(),
        )

        self.logger = logger

        # LLM state information (all turns)
        self.curr_prompt: str = "" # current context (built out of all the info from llm_info)
        self.llm_info = LLMTurnInfo()

        # timeout stuff
        self.compilation_timeout = config.timeout
        self.correctness_check_timeout = config.timeout

        # check if run is already finished
        if os.path.exists(os.path.join(self.logger.log_dir, "DONE")):
            print(
                f"[SKIP] Run {self.config.run_name}, problem id {self.work.problem_id}, sample {self.work.sample_id} already finished... skipping"
            )
            self.finished = True # skip the whole run
            return

        # check if previous run exists
        elif os.path.exists(self.logger.log_file):
            print(
                f"[RECOVER {self.work.problem_id}/{self.work.sample_id}] "
                f"Run was not finished, loading existing partial results from {self.logger.log_file}"
            )
            self.load_from_previous_run()

        self.finished = False

        # set up orchestrator
        self.process_id = process_id
        self.orchestrator = orchestrator

    def load_from_previous_run(self):
        """
        Load previous information from a prior run if it exists in the log dir.
        """
        self.logger.load_log()
        saved_log = copy.deepcopy(self.logger.current_log)

        # clean the log at this point
        # in case the run finished abruptly, we need to rebuild log
        self.logger.clean_log()

        if self.config.verbose:
            print(
                f"[RECOVER {self.work.problem_id}/{self.work.sample_id}] "
                    "Recoreved log data from previous run: ",
                saved_log.keys(),
            )

        # check turn data
        for turn in range(1, self.max_k + 2):

            # check if this is the first turn that is not recorded in the log
            self.current_k = turn
            if turn not in saved_log:
                # start from this turn
                break

            # current turn
            turn_data = saved_log[turn]

            self.llm_info.update_turn_data(turn, {
                "prompt": turn_data.get("prompt", ""),
                "model_response": turn_data.get("model_response", ""),
                "kernel_code": turn_data.get("kernel_code", ""),
                "eval_result": turn_data.get("eval_result", {}),
                "profiler_result": turn_data.get("profiler_result", ""),
            })

            # if these are empty, this turn was corrupted somehow
            # re-do this turn
            if (
                self.llm_info.prompt[turn] == ""
                or self.llm_info.model_response[turn] == ""
                or self.llm_info.kernel_code[turn] == ""
            ):
                self.current_k = turn
                break

            # otherwise, rebuild turn log data
            self.logger.update_turn(turn=turn, llm_info=self.llm_info)

        # at the end of recovery, save log
        # if nothing was wrong, then the same info is dumped; if something was
        # wrong at some round, then we write to discard any later data
        self.logger.save_log()

        # special case: everything is finished, but the DONE file is not written
        # for whatever reason; passthrough to the end
        if self.current_k == self.max_k + 1:
            self.current_k -= 1
            self.state = CaesarState.FINISH_STATE


        if self.config.verbose:
            print(
                f"[RECOVER {self.work.problem_id}/{self.work.sample_id}] "
                f"Resuming from round {self.current_k}"
            )


    def run(self) -> None:
        """
        Main state machine event loop.
        """

        # check if DONE file written
        if self.finished:
            return

        while self.current_k <= self.max_k:

            if self.config.show_state:
                print(
                    f"[STATEMACHINE {self.work.problem_id}/{self.work.sample_id}] "
                    f"Round {self.current_k}, entering state: {self.state}"
                )

            match self.state:
                case CaesarState.START_STATE:
                    self.start_turn_logic()
                case CaesarState.GENERATE_STATE:
                    self.generate_logic()
                case CaesarState.COMPILE_STATE:
                    self.compile_logic()
                case CaesarState.CORRECTNESS_STATE:
                    self.correctness_check_logic()
                case CaesarState.PERFORMANCE_STATE:
                    self.performance_logic()
                case CaesarState.FINISH_STATE:
                    self.finish_turn_logic()

                case _:
                    raise ValueError(f"Invalid state to be in: {self.state}")

            # transition to next state
            # (current_state, outcome) -> next_state
            self.state = self.transitions[self.outcome]


    def start_turn_logic(self):
        """
        Logic for the start state of each turn. If we didn't yet reach the
        maximum number of turns allowed, keep looping.
        """

        # initialize this round's prompt with the information so far
        self.curr_prompt = build_llm_prompt_for_turn(
            turn=self.current_k,
            ref_arch_src=self.ref_problem_src,
            kernels=self.llm_info.kernel_code,
            eval_result=self.llm_info.eval_result,
            profiler_result=self.llm_info.profiler_result,
            strategy=self.config.prompt_strategy,
            max_profiler_feedback_length=2000, # TODO this is in characters; how big can traces actually get? #self.config.max_feedback_length,
        )
        self.llm_info.prompt[self.current_k] = self.curr_prompt

        self.outcome = StateOutcome.Start

    def generate_logic(self):
        """
        Logic for the generation state. Query LLM given context and generate
        kernel.
        """
        # query LLM
        model_response = query_server(
            self.curr_prompt,
            model_name=self.config.model_name,
            temperature=(
                0.0 if self.config.greedy_sample else self.config.temperature
            ),
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens, # for? also should be max_tokens - prompt
            num_completions=self.config.num_completions, # for?
            server_port=self.config.server_port,
            server_address=self.config.server_address,
            server_type=self.config.server_type,
        )
        self.llm_info.model_response[self.current_k] = model_response

        kernel_code = extract_last_code(model_response, ["python", "cpp"])

        # if we failed to generate a kernel, simply move to the next round
        if kernel_code is None or len(kernel_code) == 0:
            if self.config.verbose:
                print(
                    f"[GENERATE {self.work.problem_id}/{self.work.sample_id}] "
                    "Failed to generate kernel code."
                )
            self.outcome = StateOutcome.GenerateFail
        else:
            self.llm_info.kernel_code[self.current_k] = kernel_code
            self.outcome = StateOutcome.GenerateSuccess

    def compile_logic(self):
        """
        Logic for the CPU compilation state.
        """

        # compile kernel and build cache
        returncode, stdout, err = compile_single_sample(
            kernel_src=self.llm_info.kernel_code[self.current_k],
            config=self.config,
            build_dir=self.build_dir,
            timeout_seconds=self.compilation_timeout
        )

        if self.config.verbose:
            print(f"[COMPILE {self.work.problem_id}/{self.work.sample_id}] Return code: {returncode}")
            print(f"[COMPILE {self.work.problem_id}/{self.work.sample_id}] Compile stdout: {stdout}")
            print(f"[COMPILE {self.work.problem_id}/{self.work.sample_id}] Compile stderr: {err}")

        if returncode == 0:
            # write partial eval result here, since compilation succeeded
            # we'll write more later if doing correctness check
            self.llm_info.eval_result[self.current_k] = kernel_eval.KernelExecResult(
                compiled=True,
                metadata={
                    "hardware": "cpu",
                    "device": "cpu",
                }
            )
            self.outcome = StateOutcome.CompileSuccess
        else:
            # register compilation failure as eval result
            self.llm_info.eval_result[self.current_k] = kernel_eval.KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "compiler_error": f"Compilation failed.\nstdout: {stdout}\nstderr: {err}",
                    "hardware": "cpu",
                    "device": "cpu"
                }
            )
            self.outcome = StateOutcome.CompileFail

    def correctness_check_logic(self):
        """
        Check kernel code correctness.
        """
        if self.config.verbose:
            print(f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] Requesting GPU...")

        with self.orchestrator.reserve_gpu() as gpu_id:
            try:
                if self.config.verbose:
                    print(
                        f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] "
                        f"Acquired GPU {gpu_id}"
                    )

                device = torch.device(f"cuda:{gpu_id}")
                start_time = time.time()

                # TODO look into this more
                result: kernel_eval.KernelExecResult = evaluate_single_sample_src(
                    ref_arch_src=self.ref_problem_src,
                    kernel_src=self.llm_info.kernel_code[self.current_k],
                    configs=self.config,
                    build_dir=self.build_dir,
                    timeout_seconds=self.correctness_check_timeout,
                    device=device
                )

                if self.config.verbose:
                    print(
                        f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] Result: ",
                        result,
                    )

                work_time = time.time() - start_time

                # record result (fields should be correctly set)
                self.llm_info.eval_result[self.current_k] = result

                # if compiled and is correct
                if result is not None and result.compiled and result.correctness:
                    self.outcome = StateOutcome.CorrectnessSuccess
                else:
                    self.outcome = StateOutcome.CorrectnessFail

                if self.config.verbose:
                    print(
                        f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] "
                        f"Working on GPU {gpu_id} ({device}) for {work_time:.2f} seconds"
                    )

            except TimeoutError:
                print(
                    f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] "
                    f"Working on GPU {gpu_id} operation timed out"
                )
                self.outcome = StateOutcome.CorrectnessFail
                self.llm_info.eval_result[self.current_k] = kernel_eval.KernelExecResult(
                    compiled=False,
                    correctness=False,
                    metadata={
                        "timeout_error": "GPU timed out.",
                        "hardware": "gpu",
                        "device": f"cuda:{gpu_id}"
                    }
                )

            except Exception as e:
                print(
                    f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] "
                    f"Working on GPU {gpu_id} encountered error: {e}"
                )

            finally:
                try:
                    del device
                except:
                    pass

                if self.config.verbose:
                    print(
                        f"[CORRECTNESS {self.work.problem_id}/{self.work.sample_id}] "
                        f"Released GPU {gpu_id}"
                    )

    def performance_logic(self):
        """
        Logic for the performance state. This is for profiling code.
        """

        if self.config.verbose:
            print(f"[PERF {self.work.problem_id}/{self.work.sample_id}] Requesting GPU...")

        with self.orchestrator.reserve_gpu() as gpu_id:
            try:
                print(f"[PERF {self.work.problem_id}/{self.work.sample_id}] Acquired GPU {gpu_id}")

                device = torch.device(f"cuda:{gpu_id}")
                start_time = time.time()

                # TODO look into this more
                result = get_torch_profiler_info(
                    ref_arch_src=self.ref_problem_src,
                    kernel_src=self.llm_info.kernel_code[self.current_k],
                    build_dir=self.build_dir,
                    device=device,
                )
                self.llm_info.profiler_result[self.current_k] = result

                work_time = time.time() - start_time

                if self.config.verbose:
                    print(
                        f"[PERF {self.work.problem_id}/{self.work.sample_id}] "
                        f"Working on GPU {gpu_id} ({device}) for {work_time:.2f} seconds"
                    )

            except Exception as e:
                print(
                    f"[PERF {self.work.problem_id}/{self.work.sample_id}] "
                    f"Working on GPU {gpu_id} encountered error: {e}"
                )

            finally:
                try:
                    del device
                except:
                    pass

                if self.config.verbose:
                    print(
                        f"[PERF {self.work.problem_id}/{self.work.sample_id}] "
                        f"Released GPU {gpu_id}"
                    )

        self.outcome = StateOutcome.Performance

    def finish_turn_logic(self):
        """
        Logic for the finish state of a turn.
        """

        # this is reached at the end of each round; if the round, however,
        # is not the LAST ROUND, we simply pass through to the next round's
        # start state

        # save the current round's state
        self.logger.update_turn_and_log(self.current_k, self.llm_info)

        # increment round number
        self.outcome = StateOutcome.Finish
        self.current_k += 1

        # IF last round, mark that this run is finished
        if self.current_k > self.max_k:
            if self.config.verbose:
                print(
                    f"[FINISH {self.work.problem_id}/{self.work.sample_id}] "
                    "Finished run, writing DONE file"
                )
            with open(os.path.join(self.logger.log_dir, "DONE"), "w") as _:
                pass
