import time
import os
import copy
import json
import re
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict

from KernelBenchInternal import eval as kernel_eval
from KernelBenchInternal.utils import (
    extract_last_code,
    read_file,
)
from langsmith import trace
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from eval import (
    compile_single_sample,
    evaluate_single_sample_src_mp,
    get_ncu_kernel_metrics_mp,
    top24_metrics_cuda_forge,
)
from prompt_state_machine import (
    build_llm_prompt,
    build_code_agent_system_prompt,
    PromptKernelContext,
)

from prompts import (
    COMPILE_SUMMARY_USER_INPUT,
    RUNTIME_SUMMARY_USER_INPUT,
    PROFILER_SUMMARY_USER_INPUT,
    REVIEWER_AGENT_SYSTEM_PROMPT,
    PROMPT_AGENT_SYSTEM_PROMPT,
)



from mcts import MCTSTree, compute_reward
from states import StateOutcome


from logger import CaesarLogger
from work import WorkArgs

from utils import (
    ensure_json_serializable,
    create_code_agent,
    create_prompt_agent,
    create_reviewer_agent,
    fetch_baseline_time_by_problem_id,
)

from orchestrator import GPUOrchestrator

from caesar_config import CaesarRunConfig

from conversation_info import ConversationInfo



from rag import RagIndex, build_or_load_rag_index, rag_retrieve


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
    code_agent: CompiledStateGraph
    prompt_agent: CompiledStateGraph
    reviewer_agent: CompiledStateGraph
    mcts_tree: MCTSTree
    baseline_runtime_ms: float | None
    rag_index: RagIndex



# graph state that is updated when iterating between the state machine rounds
class CaesarGraphState(TypedDict):
    conversation_info: ConversationInfo
    current_turn: int
    state_outcome: StateOutcome
    selected_node_id: int
    selected_node_path: list[int]
    current_node_id: int


def _infer_level_from_problem_path(problem_path: str) -> int | None:
    if not problem_path:
        return None
    match = re.search(r"level(\d+)", problem_path)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _infer_level_from_dataset_name(dataset_name: str) -> int | None:
    if not dataset_name:
        return None
    # Mixed-level datasets (e.g., levels12-subset) should not infer from name.
    if "levels" in dataset_name:
        return None
    match = re.search(r"level(\d+)", dataset_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _coerce_baseline_ms(baseline_entry) -> float | None:
    if isinstance(baseline_entry, (int, float)):
        return float(baseline_entry)
    if isinstance(baseline_entry, dict):
        for key in ("mean", "median", "min", "baseline_ms", "runtime_ms"):
            value = baseline_entry.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return None




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




def select_mcts_node_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Select the next MCTS node to expand.
    """
    tree = runtime.context.mcts_tree
    path, node_id = tree.select()
    state['selected_node_id'] = node_id
    state['selected_node_path'] = path
    return state



def create_prompt_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Create the prompt for the model.
    """
    config = runtime.context.config
    prompt_agent = runtime.context.prompt_agent
    work = runtime.context.work
    conv_info = state['conversation_info']
    tree = runtime.context.mcts_tree
    current_turn = state['current_turn']


    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: CREATE_PROMPT"
        )

    parent_id = state['selected_node_id']
    parent_node = tree.get_node(parent_id)

    # create a new child node for this mutation
    child_id = tree.add_child(parent_id)
    state['current_node_id'] = child_id
    child_node = tree.get_node(child_id)

    # build kernel context for prompt construction
    kernel_context = PromptKernelContext()
    best_node = tree.best_node()
    if best_node is not None and best_node.node_id != parent_id:
        if best_node.kernel_code:
            kernel_context.kernel_code[best_node.node_id] = best_node.kernel_code
        if best_node.eval_result is not None:
            kernel_context.eval_result[best_node.node_id] = best_node.eval_result
        if best_node.profiler_summary:
            kernel_context.profiler_summary[best_node.node_id] = (
                best_node.profiler_summary
            )

    if parent_node.kernel_code:
        kernel_context.kernel_code[parent_id] = parent_node.kernel_code
    if parent_node.eval_result is not None:
        kernel_context.eval_result[parent_id] = parent_node.eval_result
    if parent_node.compile_summary:
        kernel_context.compile_summary[parent_id] = parent_node.compile_summary
    if parent_node.runtime_summary:
        kernel_context.runtime_summary[parent_id] = parent_node.runtime_summary
    if parent_node.profiler_summary:
        kernel_context.profiler_summary[parent_id] = parent_node.profiler_summary

    base_prompt = build_llm_prompt(
        config=config,
        turn=current_turn,
        kernel_context=kernel_context,
    )
    conv_info.prompt_agent_input[current_turn] = base_prompt

    if base_prompt.strip() == "":
        final_prompt = ""
        child_node.prompt_messages = copy.deepcopy(parent_node.prompt_messages)
    else:
        prompt_messages = list(parent_node.prompt_messages)
        if prompt_messages and isinstance(prompt_messages[0], BaseMessage):
            prompt_messages.append(HumanMessage(content=base_prompt))
        else:
            prompt_messages.append({"role": "user", "content": base_prompt})

        prompt_context = {
            "rag_index": runtime.context.rag_index,
            "conv_info": conv_info,
            "rag_top_k": runtime.context.config.rag_top_k,
            "rag_scope": runtime.context.config.rag_scope,
            "problem_id": runtime.context.work.problem_id,
            "turn": current_turn,
        }

        response = prompt_agent.invoke({"messages": prompt_messages}, context=prompt_context)


        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            last_message: AIMessage = messages[-1]
            final_prompt = (
                getattr(last_message, "content", None)
                or getattr(last_message, "text", "")
            )
            child_node.prompt_messages = messages
        else:
            final_prompt = response.content
            child_node.prompt_messages = (
                prompt_messages + [{"role": "assistant", "content": final_prompt}]
            )

    conv_info.prompt_agent_output[current_turn] = final_prompt
    conv_info.input_prompt[current_turn] = final_prompt

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
    code_agent = runtime.context.code_agent
    current_turn = state['current_turn']
    conv_info = state['conversation_info']
    tree = runtime.context.mcts_tree
    node = tree.get_node(state['current_node_id'])


    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: QUERY_LLM"
        )

    prompt_text = conv_info.input_prompt.get(current_turn, "")
    coding_messages = list(node.coding_messages)
    if coding_messages and isinstance(coding_messages[0], BaseMessage):
        coding_messages.append(HumanMessage(content=prompt_text))
    else:
        coding_messages.append({"role": "user", "content": prompt_text})

    response = code_agent.invoke({"messages": coding_messages})

    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        last_message: AIMessage = messages[-1]
        model_content = (
            getattr(last_message, "content", None)
            or getattr(last_message, "text", "")
        )
        usage_metadata = last_message.usage_metadata
        node.coding_messages = messages
    else:
        model_content = response.content
        usage_metadata = getattr(response, "usage_metadata", {}) or {}
        node.coding_messages = (
            coding_messages + [{"role": "assistant", "content": model_content}]
        )

    conv_info.model_response[current_turn] = model_content
    conv_info.token_usage[current_turn] = usage_metadata

    kernel_code = extract_last_code(
        conv_info.model_response[current_turn], ["python", "cpp"]
    )

    if kernel_code is None or len(kernel_code) == 0:
        if config.verbose:
            print(
                f"[GENERATE {work.problem_id}/{work.sample_id}] "
                "Failed to generate kernel code."
            )
        node.kernel_code = ""
        node.reward = 0.0
        tree.backprop(state['selected_node_path'] + [node.node_id], 0.0)
        state['state_outcome'] = StateOutcome.GenerateFail
    else:
        conv_info.kernel_code[current_turn] = kernel_code
        node.kernel_code = kernel_code
        state['state_outcome'] = StateOutcome.GenerateSuccess

    return state



def compile_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Logic for the CPU compilation state.
    """
    config = runtime.context.config
    reviewer_agent = runtime.context.reviewer_agent
    tree = runtime.context.mcts_tree

    work = runtime.context.work
    current_turn = state['current_turn']
    conv_info = state['conversation_info']
    node = tree.get_node(state['current_node_id'])


    if config.show_state:
        print(
            f"[STATEMACHINE {work.problem_id}/{work.sample_id}] "
            f"Round {current_turn}, entering state: COMPILATION"
        )

    with runtime.context.worker_semaphore:
        returncode, stdout, stderr = compile_single_sample(
            kernel_src=node.kernel_code,
            gpu_arch=config.gpu_arch,
            build_dir=runtime.context.build_dir,
            timeout_seconds=config.timeout
        )

    if config.verbose:
        print(f"[COMPILE {work.problem_id}/{work.sample_id}] Return code: {returncode}")
        print(f"[COMPILE {work.problem_id}/{work.sample_id}] Compile stdout: ...{stdout[-1000:]}")
        print(f"[COMPILE {work.problem_id}/{work.sample_id}] Compile stderr: ...{stderr[-1000:]}")

    if returncode == 0:
        conv_info.eval_result[current_turn] = kernel_eval.KernelExecResult(
            compiled=True,
            metadata={
                "hardware": "cpu",
                "device": "cpu",
            }
        )
        node.eval_result = conv_info.eval_result[current_turn]
        state['state_outcome'] = StateOutcome.CompileSuccess
    else:
        comp_prompt = COMPILE_SUMMARY_USER_INPUT.format(
            kernel_code=node.kernel_code,
            stdout=stdout[-100_000:],
            stderr=stderr[-100_000:],
        )
        conv_info.compile_prompt[current_turn] = comp_prompt
        reviewer_response = reviewer_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": comp_prompt,
                }
            ]
        })

        last_message: AIMessage = reviewer_response["messages"][-1]
        reviewer_content = (
            getattr(last_message, "content", None)
            or getattr(last_message, "text", "")
        )
        reviewer_usage = last_message.usage_metadata

        summary = {
            "content": reviewer_content,
            "token_usage": reviewer_usage,
        }
        conv_info.compile_summary[current_turn] = summary
        node.compile_summary = summary

        conv_info.eval_result[current_turn] = kernel_eval.KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={
                "compiler_error": f"Compilation failed.\nstdout: {stdout}\nstderr: {stderr}",
                "hardware": "cpu",
                "device": "cpu"
            }
        )
        node.eval_result = conv_info.eval_result[current_turn]

        reward = compute_reward(node.eval_result, runtime.context.baseline_runtime_ms)
        node.reward = reward
        tree.backprop(state['selected_node_path'] + [node.node_id], reward)

        state['state_outcome'] = StateOutcome.CompileFail
    return state



def correctness_check_handler(
    state: CaesarGraphState, runtime: Runtime[CaesarRuntimeContext]
) -> CaesarGraphState:
    """
    Check kernel code correctness.
    """
    config = runtime.context.config
    reviewer_agent = runtime.context.reviewer_agent
    orchestrator = runtime.context.orchestrator
    work = runtime.context.work
    ref_problem_src = runtime.context.ref_problem_src
    current_turn = state['current_turn']
    conv_info = state['conversation_info']
    tree = runtime.context.mcts_tree
    node = tree.get_node(state['current_node_id'])

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

        result_queue = mp.Queue()
        proc = mp.Process(
            target=evaluate_single_sample_src_mp,
            args=(
                ref_problem_src,
                node.kernel_code,
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
            proc.terminate()
            print(
                f"[CORRECTNESS {work.problem_id}/{work.sample_id}] "
                f"Working on GPU {gpu_id} operation timed out."
            )
            state['state_outcome'] = StateOutcome.CorrectnessFail
            conv_info.eval_result[current_turn] = kernel_eval.KernelExecResult(
                compiled=True,
                correctness=False,
                metadata={
                    "timeout_error": "GPU timed out.",
                    "hardware": "gpu",
                    "device": f"cuda:{gpu_id}"
                }
            )
            node.eval_result = conv_info.eval_result[current_turn]
        else:
            result = result_queue.get()

            if config.verbose:
                print(
                    f"[CORRECTNESS {work.problem_id}/{work.sample_id}] Result: ",
                    result,
                )

            result.compiled = True
            conv_info.eval_result[current_turn] = result
            node.eval_result = result

            if result is not None and result.compiled and result.correctness:
                state['state_outcome'] = StateOutcome.CorrectnessSuccess
            else:
                meta = result.metadata.get("correctness_issue", "")
                if meta == "":
                    meta = result.metadata.get("cuda_error", "")
                if meta == "":
                    meta = result.metadata.get("timeout_error", "")
                if meta == "":
                    meta = result.metadata.get("other_error", "")

                run_prompt = RUNTIME_SUMMARY_USER_INPUT.format(
                    kernel_code=node.kernel_code,
                    metadata=meta,
                )
                conv_info.runtime_prompt[current_turn] = run_prompt
                reviewer_response = reviewer_agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": run_prompt,
                        }
                    ]
                })
                last_message: AIMessage = reviewer_response["messages"][-1]
                reviewer_content = (
                    getattr(last_message, "content", None)
                    or getattr(last_message, "text", "")
                )
                reviewer_usage = last_message.usage_metadata

                summary = {
                    "content": reviewer_content,
                    "token_usage": reviewer_usage,
                }
                conv_info.runtime_summary[current_turn] = summary
                node.runtime_summary = summary

                state['state_outcome'] = StateOutcome.CorrectnessFail

            if config.verbose:
                print(
                    f"[CORRECTNESS {work.problem_id}/{work.sample_id}] "
                    f"Working on GPU {gpu_id} for {work_time:.2f} seconds"
                )

        reward = compute_reward(node.eval_result, runtime.context.baseline_runtime_ms)
        node.reward = reward
        tree.backprop(state['selected_node_path'] + [node.node_id], reward)

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
    reviewer_agent = runtime.context.reviewer_agent

    orchestrator = runtime.context.orchestrator
    work = runtime.context.work
    ref_problem_src = runtime.context.ref_problem_src
    current_turn = state['current_turn']
    conv_info = state['conversation_info']
    tree = runtime.context.mcts_tree
    node = tree.get_node(state['current_node_id'])


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

        start_time = time.time()
        result_queue = mp.Queue()
        proc = mp.Process(
            target=get_ncu_kernel_metrics_mp,
            args=(
                ref_problem_src,
                node.kernel_code,

                runtime.context.build_dir,
                gpu_id,
                config.num_perf_trials,
                top24_metrics_cuda_forge,
                42,
                result_queue,

            ),
        )
        proc.start()
        proc.join() # wait forever for profiler
        work_time = time.time() - start_time
        result = result_queue.get()

        conv_info.profiler_result[current_turn] = result

        prof_prompt = PROFILER_SUMMARY_USER_INPUT.format(
            kernel_code=node.kernel_code,
            profiler_output=result,
        )
        conv_info.profiler_prompt[current_turn] = prof_prompt
        reviewer_response = reviewer_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": prof_prompt,
                }
            ]
        })
        last_message: AIMessage = reviewer_response["messages"][-1]
        reviewer_content = (
            getattr(last_message, "content", None)
            or getattr(last_message, "text", "")
        )
        reviewer_usage = last_message.usage_metadata

        summary = {
            "content": reviewer_content,
            "token_usage": reviewer_usage,
        }
        conv_info.profiler_summary[current_turn] = summary
        node.profiler_summary = summary



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
    builder = StateGraph(CaesarGraphState, context_schema=CaesarRuntimeContext)

    # init
    builder.add_node('setup_state_machine_handler', setup_state_machine_handler)

    # actual machine states
    builder.add_node('select_mcts_node_handler', select_mcts_node_handler)
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
            StateOutcome.SetupDone: 'select_mcts_node_handler',
            StateOutcome.SetupFinishRun: END
        }
    )
    builder.add_edge('select_mcts_node_handler', 'create_prompt_handler')
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
        lambda state, runtime:
            'performance_handler'
            if (
                state['state_outcome'] == StateOutcome.CorrectnessSuccess
                and state['current_turn'] < runtime.context.config.max_turn
            ) else
            'finish_turn_handler',
        ['performance_handler', 'finish_turn_handler']
    )
    builder.add_edge('performance_handler', 'finish_turn_handler')
    builder.add_conditional_edges(
        'finish_turn_handler',
        lambda state: state['state_outcome'],
        {
            StateOutcome.NextTurn: 'select_mcts_node_handler',
            StateOutcome.EndRun: END
        }
    )

    return builder.compile()



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

        ref_problem_src = read_file(work.problem_path)

        # build the rag index
        rag_index = build_or_load_rag_index(
            docs_dir=config.rag_docs_dir,
            index_dir=config.rag_index_dir,
            manifest_path=config.rag_manifest_path,
        )

        mcts_tree = MCTSTree(
            max_children=config.mcts_max_children,
            exploration=config.mcts_exploration,
        )
        baseline_runtime_ms: float | None = None
        baseline_time_path = None
        if getattr(config, "timing_baseline_dir", None) and getattr(
            config, "timing_baseline_filename", None
        ):
            baseline_time_path = os.path.join(
                config.timing_baseline_dir, config.timing_baseline_filename
            )
        elif getattr(config, "timing_baseline_path", None):
            baseline_time_path = config.timing_baseline_path

        if baseline_time_path:
            level = _infer_level_from_problem_path(work.problem_path)
            if level is None:
                level = _infer_level_from_dataset_name(config.dataset_name)
            if level is None:
                if config.verbose:
                    print(
                        "[TIMING] Could not infer dataset level from problem "
                        f"path {work.problem_path} or dataset "
                        f"{config.dataset_name}; skipping baseline timing."
                    )
            else:
                try:
                    baseline_entry = fetch_baseline_time_by_problem_id(
                        baseline_time_filepath=baseline_time_path,
                        level=level,
                        problem_id=work.problem_id,
                    )
                    baseline_runtime_ms = _coerce_baseline_ms(baseline_entry)
                    if baseline_runtime_ms is None and config.verbose:
                        print(
                            "[TIMING] Baseline timing entry missing numeric "
                            f"value for problem {work.problem_id} in "
                            f"{baseline_time_path}."
                        )
                except FileNotFoundError as exc:
                    if config.verbose:
                        print(f"[TIMING] {exc}")
                except AssertionError as exc:
                    if config.verbose:
                        print(f"[TIMING] {exc}")



        # build the code agent system prompt (includes examples + ref kernel)
        code_agent_system_prompt = build_code_agent_system_prompt(
            config=config,
            ref_arch_src=ref_problem_src,
        )


        code_agent = create_code_agent(
            **base_llm_opts,
            system_prompt=code_agent_system_prompt,
        )
        reviewer_agent = create_reviewer_agent(**base_llm_opts)

        prompt_agent_system_prompt = PROMPT_AGENT_SYSTEM_PROMPT.format(
            max_turn=config.max_turn,
        )
        prompt_agent = create_prompt_agent(
            **base_llm_opts,
            system_prompt=prompt_agent_system_prompt,
            tools=[rag_retrieve],
        )

        # save the initial system prompts
        conv_info = ConversationInfo(
            coding_agent_system_prompt=code_agent_system_prompt,
            prompt_agent_system_prompt=prompt_agent_system_prompt,
            reviewer_agent_system_prompt=REVIEWER_AGENT_SYSTEM_PROMPT,
        )



        graph = _init_state_machine_graph()

        # initialize state setup
        initial_context: CaesarRuntimeContext = {
            'process_id': process_id,
            'config': config,
            'work': work,

            # contains the reference problem in Python code as a string;
            # load it from KernelBench repo
            'ref_problem_src': ref_problem_src,
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
            'code_agent': code_agent,
            'prompt_agent': prompt_agent,
            'reviewer_agent': reviewer_agent,
            'mcts_tree': mcts_tree,
            'baseline_runtime_ms': baseline_runtime_ms,
            'rag_index': rag_index,

        }

        initial_state: CaesarGraphState = {
            'conversation_info': conv_info,
            'current_turn': 1,
            'state_outcome': StateOutcome.EndRun,
            'selected_node_id': mcts_tree.root_id,
            'selected_node_path': [mcts_tree.root_id],
            'current_node_id': mcts_tree.root_id,
        }


        # launch graph
        with trace(name=f'problem-{work.problem_id}-sample-{work.sample_id}'):
            graph.invoke(
                initial_state, {"recursion_limit": 1000}, context=initial_context
            )

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
    for sample in range(config.num_samples):

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
        print(f"State machine worker {os.getpid()} finished work for problem {work.problem_id}")
