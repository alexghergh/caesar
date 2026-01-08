import time
import os
import copy
import json
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.messages import AIMessage

from KernelBenchInternal import eval as kernel_eval
from KernelBenchInternal.utils import (
    extract_last_code,
    read_file,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langsmith import trace

from eval import (
    compile_single_sample,
    evaluate_single_sample_src_mp,
    get_torch_profiler_info_mp,
)
from prompt_state_machine import build_llm_prompt
from prompts import (
    COMPILE_SUMMARY_SYSTEM_PROMPT,
    COMPILE_SUMMARY_USER_INPUT,
    RUNTIME_SUMMARY_SYSTEM_PROMPT,
    RUNTIME_SUMMARY_USER_INPUT,
    PROFILER_SUMMARY_SYSTEM_PROMPT,
    PROFILER_SUMMARY_USER_INPUT,
)
from states import StateOutcome
from work import WorkArgs
from logger import CaesarLogger
from utils import ensure_json_serializable, create_llm
from orchestrator import GPUOrchestrator
from caesar_config import CaesarRunConfig
from conversation_info import ConversationInfo


# context that doesn't change during state machine traversal
@dataclass
class CaesarRuntimeContext:
    process_id: int
    config: CaesarRunConfig
    work: WorkArgs
    ref_problem_src: str
    logger: CaesarLogger
    build_dir: str | os.PathLike
    orchestrator: GPUOrchestrator
    worker_semaphore: mp.Semaphore
    code_llm: CompiledStateGraph
    prompt_llm: CompiledStateGraph
    summary_llm: CompiledStateGraph


# graph state that is updated when iterating between the state machine rounds
class CaesarGraphState(TypedDict):
    conversation_info: ConversationInfo
    current_turn: int
    state_outcome: StateOutcome


def setup_state_machine_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Initialize all required fields from the graph state.
    """
    logger = runtime.context.logger
    config = runtime.context.config
    work = runtime.context.work
    conversation_info = state['conversation_info']

    # skip if run already finished
    if os.path.exists(os.path.join(logger.log_dir, "DONE")):
        print(
            f"[SKIP] Run {config.run_name}, problem id {work.problem_id}, "
            f"sample id {work.sample_id} already finished... skipping"
        )
        state['state_outcome'] = StateOutcome.SetupFinishRun
        return state

    # resume log info if available
    if os.path.exists(logger.log_file):
        print(
            f"[RECOVER {work.problem_id}/{work.sample_id}] "
            f"Run was not finished, loading existing partial results from "
            f"{logger.log_file}"
        )

        logger.load_log()
        saved_log = copy.deepcopy(logger.current_log)

        # clean the log at this point
        # in case the run finished abruptly, we need to rebuild log
        logger.clean_log()

        if config.verbose:
            print(
                f"[RECOVER {work.problem_id}/{work.sample_id}] "
                    "Recoreved log data from previous run: ",
                saved_log.keys(),
            )

        # check turn data
        for turn in range(1, config.max_turn + 2):

            # check if this is the first turn that is not recorded in the log
            state['current_turn'] = turn
            if turn not in saved_log:
                # start from this turn
                break

            # current turn
            turn_data = saved_log[turn]

            # update turn data
            conversation_info.update_turn_data(turn, turn_data)

            # if these are empty, this turn was corrupted somehow
            # re-do this turn
            if (
                conversation_info.input_prompt[turn] == ""
                or conversation_info.model_response[turn] == ""
                or conversation_info.kernel_code[turn] == ""
            ):
                state['current_turn'] = turn
                break

            # otherwise, rebuild turn log data
            logger.update_turn(turn=turn, llm_info=conversation_info)

        # at the end of recovery, save log
        # if nothing was wrong, then the same info is dumped; if something was
        # wrong at some round, then we write to discard any later data
        logger.save_log()

        # special case: everything is finished, but the DONE file is not written
        # for whatever reason; passthrough to the end
        if state['current_turn'] == config.max_turn + 1:
            state['current_turn'] -= 1
            state['state_outcome'] = StateOutcome.SetupFinishRun
            return state

        if config.verbose:
            print(
                f"[RECOVER {work.problem_id}/{work.sample_id}] "
                f"Resuming from round {state['current_turn']}"
            )

    state['state_outcome'] = StateOutcome.SetupDone
    return state


def create_prompt_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Create the prompt for the model.
    """
    config = runtime.context.config
    prompt_llm = runtime.context.prompt_llm
    work = runtime.context.work
    ref_problem_src = runtime.context.ref_problem_src
    conv_info = state['conversation_info']
    current_turn = state['current_turn']

    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: CREATE_PROMPT"
        )

    # initialize this round's prompt with the information so far
    conv_info.input_prompt[current_turn] = build_llm_prompt(
        config=config,
        prompt_llm=prompt_llm,
        turn=current_turn,
        ref_arch_src=ref_problem_src,
        conv_info=conv_info,
    )

    state['state_outcome'] = StateOutcome.Start
    return state


def query_llm_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Logic for the generation state. Query LLM given context and generate kernel.
    """
    config = runtime.context.config
    work = runtime.context.work
    code_llm = runtime.context.code_llm
    current_turn = state['current_turn']
    conv_info = state['conversation_info']

    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: QUERY_LLM"
        )

    # query LLM
    model_response: AIMessage = code_llm.invoke(conv_info.input_prompt[current_turn])

    conv_info.model_response[current_turn] = model_response.content
    conv_info.token_usage[current_turn] = model_response.usage_metadata

    kernel_code = extract_last_code(
        conv_info.model_response[current_turn], ["python", "cpp"]
    )

    # if we failed to generate a kernel, simply move to the next round
    if kernel_code is None or len(kernel_code) == 0:
        if config.verbose:
            print(
                f"[GENERATE {work.problem_id}/{work.sample_id}] "
                "Failed to generate kernel code."
            )
        state['state_outcome'] = StateOutcome.GenerateFail
    else:
        conv_info.kernel_code[current_turn] = kernel_code
        state['state_outcome'] = StateOutcome.GenerateSuccess

    return state


def compile_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Logic for the CPU compilation state.
    """
    config = runtime.context.config
    summary_llm = runtime.context.summary_llm
    work = runtime.context.work
    current_turn = state['current_turn']
    conv_info = state['conversation_info']

    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: COMPILATION"
        )

    with runtime.context.worker_semaphore:
        # compile kernel and build cache
        returncode, stdout, stderr = compile_single_sample(
            kernel_src=conv_info.kernel_code[current_turn],
            gpu_arch=config.gpu_arch,
            build_dir=runtime.context.build_dir,
            timeout_seconds=config.timeout
        )

    if config.verbose:
        print(f"[COMPILE {work.problem_id}/{work.sample_id}] Return code: {returncode}")
        print(f"[COMPILE {work.problem_id}/{work.sample_id}] Compile stdout: ...{stdout[-1000:]}")
        print(f"[COMPILE {work.problem_id}/{work.sample_id}] Compile stderr: ...{stderr[-1000:]}")

    if returncode == 0:
        # write partial eval result here, since compilation succeeded
        # we'll write more later if doing correctness check
        conv_info.eval_result[current_turn] = kernel_eval.KernelExecResult(
            compiled=True,
            metadata={
                "hardware": "cpu",
                "device": "cpu",
            }
        )
        state['state_outcome'] = StateOutcome.CompileSuccess
    else:
        # summarize the relevant parts of the output; this should curb
        # over-verbose output from the compiler on some error types
        compile_summary = summary_llm.invoke(
            [
                {"role": "system", "content": COMPILE_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMPILE_SUMMARY_USER_INPUT.format(
                        kernel_code=conv_info.kernel_code[current_turn],
                        stdout=stdout,
                        stderr=stderr,
                    ),
                },
            ]
        )
        conv_info.compile_summary[current_turn] = {
            "content": compile_summary.content,
            "token_usage": compile_summary.usage_metadata,
        }

        # register compilation failure as eval result
        conv_info.eval_result[current_turn] = kernel_eval.KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                "compiler_error": f"Compilation failed.\nstdout: {stdout}\nstderr: {stderr}",
                "hardware": "cpu",
                "device": "cpu"
            }
        )
        state['state_outcome'] = StateOutcome.CompileFail
    return state


def correctness_check_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Check kernel code correctness.
    """
    config = runtime.context.config
    summary_llm = runtime.context.summary_llm
    orchestrator = runtime.context.orchestrator
    work = runtime.context.work
    ref_problem_src = runtime.context.ref_problem_src
    current_turn = state['current_turn']
    conv_info = state['conversation_info']

    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: CORRECTNESS_CHECK"
        )

    if config.verbose:
        print(f"[CORRECTNESS {work.problem_id}/{work.sample_id}] Requesting GPU...")

    with orchestrator.reserve_gpu() as gpu_id:
        if config.verbose:
            print(
                f"[CORRECTNESS {work.problem_id}/{work.sample_id}] "
                f"Acquired GPU {gpu_id}"
            )

        # launch a separate process to do the GPU work, as each process
        # creates a pytorch context on the GPU; we want to avoid each
        # CPU worker having such a separate context that persists, so
        # spawning a separate process will clear the cache when the
        # process finishes
        result_queue = mp.Queue()
        proc = mp.Process(
            target=evaluate_single_sample_src_mp,
            args=(
                ref_problem_src,
                conv_info.kernel_code[current_turn],
                config,
                runtime.context.build_dir,
                gpu_id,
                config.timeout,
                result_queue,
            ),
        )
        start_time = time.time()
        proc.start()
        proc.join(timeout=config.timeout)
        work_time = time.time() - start_time

        if proc.is_alive():
            # this means we reached timeout
            proc.terminate()
            print(
                f"[CORRECTNESS {work.problem_id}/{work.sample_id}] "
                f"Working on GPU {gpu_id} operation timed out."
            )
            state['state_outcome'] = StateOutcome.CorrectnessFail
            conv_info.eval_result[current_turn] = kernel_eval.KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "timeout_error": "GPU timed out.",
                    "hardware": "gpu",
                    "device": f"cuda:{gpu_id}"
                }
            )
        else:
            result = result_queue.get()

            if config.verbose:
                print(
                    f"[CORRECTNESS {work.problem_id}/{work.sample_id}] Result: ",
                    result,
                )

            # record result (fields should be correctly set)
            conv_info.eval_result[current_turn] = result

            # if compiled and is correct
            if result is not None and result.compiled and result.correctness:
                state['state_outcome'] = StateOutcome.CorrectnessSuccess
            else:
                # summarize the correctness error to aid in the next round
                runtime_summary = summary_llm.invoke(
                    [
                        {"role": "system", "content": RUNTIME_SUMMARY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": RUNTIME_SUMMARY_USER_INPUT.format(
                                kernel_code=conv_info.kernel_code[current_turn],
                                metadata=result.metadata['correctness_issue']
                            ),
                        },
                    ]
                )
                conv_info.runtime_summary[current_turn] = {
                    "content": runtime_summary.content,
                    "token_usage": runtime_summary.usage_metadata,
                }

                state['state_outcome'] = StateOutcome.CorrectnessFail

            if config.verbose:
                print(
                    f"[CORRECTNESS {work.problem_id}/{work.sample_id}] "
                    f"Working on GPU {gpu_id} for {work_time:.2f} seconds"
                )

        if config.verbose:
            print(
                f"[CORRECTNESS {work.problem_id}/{work.sample_id}] "
                f"Released GPU {gpu_id}"
            )
    return state


def performance_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Logic for the performance state. This is for profiling code.
    """
    config = runtime.context.config
    summary_llm = runtime.context.summary_llm
    orchestrator = runtime.context.orchestrator
    work = runtime.context.work
    ref_problem_src = runtime.context.ref_problem_src
    current_turn = state['current_turn']
    conv_info = state['conversation_info']

    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: PROFILING"
        )

    if config.verbose:
        print(f"[PERF {work.problem_id}/{work.sample_id}] Requesting GPU...")

    with orchestrator.reserve_gpu() as gpu_id:
        if config.verbose:
            print(f"[PERF {work.problem_id}/{work.sample_id}] Acquired GPU {gpu_id}")

        # launch a separate process to do the GPU work, as each process
        # creates a pytorch context on the GPU; we want to avoid each
        # CPU worker having such a separate context that persists, so
        # spawning a separate process will clear the cache when the
        # process finishes
        result_queue = mp.Queue()
        proc = mp.Process(
            target=get_torch_profiler_info_mp,
            args=(
                ref_problem_src,
                conv_info.kernel_code[current_turn],
                runtime.context.build_dir,
                gpu_id,
                result_queue,
            ),
        )
        start_time = time.time()
        proc.start()
        proc.join() # wait forever for profiler
        work_time = time.time() - start_time
        result = result_queue.get()

        conv_info.profiler_result[current_turn] = result

        # summarize the profiler output
        profiler_summary = summary_llm.invoke(
            [
                {"role": "system", "content": PROFILER_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PROFILER_SUMMARY_USER_INPUT.format(
                        kernel_code=conv_info.kernel_code[current_turn],
                        profiler_output=result,
                    ),
                },
            ]
        )
        conv_info.profiler_summary[current_turn] = {
            "content": profiler_summary.content,
            "token_usage": profiler_summary.usage_metadata,
        }

        if config.verbose:
            print(
                f"[PERF {work.problem_id}/{work.sample_id}] "
                f"Working on GPU {gpu_id} for {work_time:.2f} seconds"
            )
            print(
                f"[PERF {work.problem_id}/{work.sample_id}] "
                f"Released GPU {gpu_id}"
            )

    state['state_outcome'] = StateOutcome.Performance
    return state


def finish_turn_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Logic for the finish state of a turn.
    """
    config = runtime.context.config
    logger = runtime.context.logger
    work = runtime.context.work
    current_turn = state['current_turn']
    conv_info = state['conversation_info']

    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: FINISH"
        )

    # this is reached at the end of each round; if the round, however,
    # is not the LAST ROUND, we simply pass through to the next round's
    # start state

    # save the current round's state
    logger.update_turn_and_log(current_turn, conv_info)

    # increment round number
    state['current_turn'] += 1
    state['state_outcome'] = StateOutcome.NextTurn

    # IF last round, mark that this run is finished
    if current_turn > config.max_turn:
        if config.verbose:
            print(
                f"[FINISH {work.problem_id}/{work.sample_id}] "
                "Finished run, writing DONE file"
            )
        with open(os.path.join(logger.log_dir, "DONE"), "w") as _:
            pass
        state['state_outcome'] = StateOutcome.EndRun

    return state


def _init_state_machine_graph() -> CompiledStateGraph:
    """
    Initialize langgraph state graph.
    """
    # set up langgraph state and graph transitions
    builder = StateGraph(CaesarGraphState, context_schema=CaesarRuntimeContext)

    # init
    builder.add_node('setup_state_machine_handler', setup_state_machine_handler)

    # actual machine states
    builder.add_node('create_prompt_handler', create_prompt_handler)
    builder.add_node('query_llm_handler', query_llm_handler)
    builder.add_node('compile_handler', compile_handler)
    builder.add_node('correctness_check_handler', correctness_check_handler)
    builder.add_node('performance_handler', performance_handler)
    builder.add_node('finish_turn_handler', finish_turn_handler)

    # transitions
    builder.add_edge(START, 'setup_state_machine_handler')

    builder.add_conditional_edges(
        'setup_state_machine_handler',
        lambda state: state['state_outcome'],
        {
            StateOutcome.SetupDone: 'create_prompt_handler',
            StateOutcome.SetupFinishRun: END
        }
    )
    builder.add_edge('create_prompt_handler', 'query_llm_handler')
    builder.add_conditional_edges(
        'query_llm_handler',
        lambda state: state['state_outcome'],
        {
            StateOutcome.GenerateSuccess: 'compile_handler',
            StateOutcome.GenerateFail: 'finish_turn_handler'
        }
    )
    builder.add_conditional_edges(
        'compile_handler',
        lambda state: state['state_outcome'],
        {
            StateOutcome.CompileSuccess: 'correctness_check_handler',
            StateOutcome.CompileFail: 'finish_turn_handler'
        }
    )
    builder.add_conditional_edges(
        'correctness_check_handler',
        lambda state: state['state_outcome'],
        {
            StateOutcome.CorrectnessSuccess: 'performance_handler',
            StateOutcome.CorrectnessFail: 'finish_turn_handler'
        }
    )
    builder.add_edge('performance_handler', 'finish_turn_handler')
    builder.add_conditional_edges(
        'finish_turn_handler',
        lambda state: state['state_outcome'],
        {
            StateOutcome.NextTurn: 'create_prompt_handler',
            StateOutcome.EndRun: END
        }
    )

    # compile graph
    graph = builder.compile()

    # save an image of the graph's state
    # print(graph.get_graph().draw_mermaid())
    # graph.get_graph().draw_mermaid_png(output_file_path='state_machine_graph.png')
    return graph


def init_and_run_graph(
    config: CaesarRunConfig,
    work: WorkArgs,
    process_id: int,
    orchestrator: GPUOrchestrator,
    progress: mp.Value,
    worker_semaphore: mp.Semaphore,
):
    try:
        base_llm_opts = {
            # sampling
            'temperature': (
                0.0 if config.greedy_sample else config.temperature
            ),
            'top_p': config.top_p,
            'top_k': config.top_k,
            'max_tokens': config.max_tokens,

            # reasoning models
            'use_reasoning_model': config.reasoning_model, # claude, gpt, gemini
            'reasoning_effort': config.reasoning_effort, # gpt-5 only
            'budget_tokens': config.reasoning_budget_tokens, # claude

            # server type
            'server_port': config.server_port,
            'server_address': config.server_address,
            'server_type': config.server_type,
            'model_name': config.model_name,
        }

        # get llms (these opts may be different in the future)
        code_llm = create_llm(**base_llm_opts)
        prompt_llm = create_llm(**base_llm_opts)
        summary_llm = create_llm(**base_llm_opts)

        # build graph
        graph = _init_state_machine_graph()

        # initialize state setup
        initial_context: CaesarRuntimeContext = {
            'process_id': process_id,
            'config': config,
            'work': work,

            # contains the reference problem in Python code as a string;
            # load it from KernelBench repo
            'ref_problem_src': read_file(work.problem_path),
            'logger': CaesarLogger(
                os.path.join(
                    config.log_dir_prefix,
                    config.run_group,
                    config.run_name,
                    work.get_log_path(),
                ),
            ),
            # build dir to cache compiled problems
            'build_dir': os.path.join(
                config.build_dir_prefix,
                config.run_group,
                config.run_name,
                work.get_log_path(),
            ),
            'orchestrator': orchestrator,
            'worker_semaphore': worker_semaphore,
            'code_llm': code_llm,
            'prompt_llm': prompt_llm,
            'summary_llm': summary_llm,
        }
        initial_state: CaesarGraphState = {
            'conversation_info': ConversationInfo(),
            'current_turn': 1,
            'state_outcome': StateOutcome.EndRun,
        }

        # launch graph
        with trace(name=f'problem-{work.problem_id}-sample-{work.sample_id}'):
            graph.invoke(initial_state, {"recursion_limit": 1000}, context=initial_context)

    finally:
        # update global progress (for each finished sample)
        with progress.get_lock():
            progress.value += 1


def run_state_machine(
    process_id: int,
    config: CaesarRunConfig,
    workargs: WorkArgs,
    orchestrator: GPUOrchestrator,
    progress: mp.Value,
    worker_semaphore: mp.Semaphore,
):
    # TODO depending on what I eventually want to do (e.g. best of k
    # kernel), the following code should change; the simple and
    # sensible thing to do is to run a turn, and then return control to
    # this state machine; after that, this state machine can take a
    # decision on whether it should exchange information between
    # trajectories or not etc.

    # log config with initial params
    log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(ensure_json_serializable(config.to_dict()), f, indent=2)

    # launch all samples on different sub-processes and wait for completion
    sample_proc_list = []
    for sample in range(0, config.num_samples):

        # create separate work for each sample
        work = copy.deepcopy(workargs)
        work.sample_id = sample

        if config.verbose:
            print(f"State machine worker {os.getpid()} starting work {work}")

        sample_proc = mp.Process(
            target=init_and_run_graph,
            args=(config, work, process_id, orchestrator, progress, worker_semaphore),
        )
        sample_proc.start()
        sample_proc_list.append(sample_proc)

    # wait for processes to finish
    for sample_proc in sample_proc_list:
        sample_proc.join()

        if config.verbose:
            print(f"State machine worker {os.getpid()} finished work {work}")
