import time
import os
import copy
import json
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, TypedDict

from langchain.agents.middleware import TodoListMiddleware, before_model, wrap_model_call
from langchain_core.language_models.chat_models import BaseChatModel

from KernelBenchInternal import eval as kernel_eval
from KernelBenchInternal.utils import (
    extract_last_code,
    read_file,
)
from langchain.tools import ToolRuntime, tool
from langchain.agents import AgentState, create_agent
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
from prompts import CODE_GENERATION_LLM_DESCRIPTION, COMPILER_TOOL_DESCRIPTION, CORRECTNESS_TOOL_DESCRIPTION, PROFILER_TOOL_DESCRIPTION, PROJECT_PLANNER_AGENT_SYSTEM_PROMPT, PROMPT_LLM_SYSTEM_PROMPT
from states import StateOutcome
from work import WorkArgs
from logger import CaesarLogger
from utils import get_llm_client
from orchestrator import GPUOrchestrator
from caesar_config import CaesarRunConfig
from conversation_info import ConversationInfo


##
## STATE
##

# agent context that doesn't change during execution
@dataclass
class CaesarRuntimeContext:
    process_id: int
    config: CaesarRunConfig
    work: WorkArgs
    logger: CaesarLogger
    ref_problem_src: str
    build_dir: str | os.PathLike
    orchestrator: GPUOrchestrator
    worker_semaphore: mp.Semaphore
    code_llm: BaseChatModel
    prompt_llm: BaseChatModel


##
## TOOLS
##

@tool(
    "code_compiler",
    description=COMPILER_TOOL_DESCRIPTION,
)
def compile(
    kernel_code: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> dict[str, str]:
    print("------ called compile")
    config = runtime.context.config
    semaphore = runtime.context.worker_semaphore

    with semaphore:
        # compile kernel and build cache
        returncode, stdout, err = compile_single_sample(
            kernel_src=kernel_code,
            gpu_arch=config.gpu_arch,
            build_dir=runtime.context.build_dir,
            timeout_seconds=config.timeout
        )

    # TODO call LLM to format, then simply return a single string with the
    # output result + formatted stdout/stderr

    return {
        "ret_code": str(returncode),
        "std out": stdout,
        "std err": err,
    }


@tool(
    "correctness_check",
    description=CORRECTNESS_TOOL_DESCRIPTION,
)
def run_kernel(
    kernel_code: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> str:
    print("------ called correctness")
    config = runtime.context.config
    orchestrator = runtime.context.orchestrator
    reference_problem_python_code = runtime.context.ref_problem_src

    with orchestrator.reserve_gpu() as gpu_id:

        # launch a separate process to do the GPU work, as each process
        # creates a pytorch context on the GPU; we want to avoid each
        # CPU worker having such a separate context that persists, so
        # spawning a separate process will clear the cache when the
        # process finishes
        result_queue = mp.Queue()
        proc = mp.Process(
            target=evaluate_single_sample_src_mp,
            args=(
                reference_problem_python_code,
                kernel_code,
                config,
                runtime.context.build_dir,
                gpu_id,
                config.timeout,
                result_queue,
            ),
        )
    return result_queue.get()


@tool(
    "code_profiler",
    description=PROFILER_TOOL_DESCRIPTION,
)
def profile_kernel(
    kernel_code: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> str:
    print("------ called profiler")
    orchestrator = runtime.context.orchestrator
    reference_problem_python_code = runtime.context.ref_problem_src

    with orchestrator.reserve_gpu() as gpu_id:
        # launch a separate process to do the GPU work, as each process
        # creates a pytorch context on the GPU; we want to avoid each
        # CPU worker having such a separate context that persists, so
        # spawning a separate process will clear the cache when the
        # process finishes
        result_queue = mp.Queue()
        proc = mp.Process(
            target=get_torch_profiler_info_mp,
            args=(
                reference_problem_python_code,
                kernel_code,
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

    return result


##
## AGENTS
##

@tool(
    "code_generator_llm",
    description=CODE_GENERATION_LLM_DESCRIPTION,
)
def generate_kernel(
    guidance: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> str:
    print("------ called generate kernel")
    config = runtime.context.config
    ref_problem_src = runtime.context.ref_problem_src
    code_llm = runtime.context.code_llm
    prompt_llm = runtime.context.prompt_llm

    # get the example prompt
    input_prompt = build_llm_prompt(
        config=config,
        turn=1,
        ref_arch_src=ref_problem_src,
        kernels={},
        eval_result={},
        profiler_result={},
        max_profiler_feedback_length=4000,  # TODO this is in characters; how big can traces actually get? #self.config.max_feedback_length,
    )

    # get a better suited prompt, modified by the LLM
    formatted_prompt = prompt_llm.invoke([{
        "role": "system",
        "content": PROMPT_LLM_SYSTEM_PROMPT,
    }, {
        "role": "user",
        "content": f"Human written prompt:\n{input_prompt}\n\nGuidance from project planner:\n{guidance}"
    }]).content

    # print("----- formatted prompt", formatted_prompt)

    kernel_code = code_llm.invoke(formatted_prompt).content
    kernel_code = extract_last_code(kernel_code, ["python", "cpp"])

    # print("--------- generated kernel: ", kernel_code)

    return kernel_code


def get_agents(
    config: CaesarRunConfig,
) -> Tuple[CompiledStateGraph, BaseChatModel, BaseChatModel]:
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

    # get base LLM
    base_model = get_llm_client(**base_llm_opts)

    # create the planner agent
    planner_agent = create_agent(
        model=base_model,
        system_prompt=PROJECT_PLANNER_AGENT_SYSTEM_PROMPT,
        tools=[
            # sub-agents
            generate_kernel,

            # tools
            compile, run_kernel, profile_kernel
        ],
        middleware=[
            # # add summarization so context doesn't overflow everything
            # SummarizationMiddleware(
            #     model=base_model, # TODO smaller model
            #     trigger=('tokens', int(max_tokens * 0.8)),
            #     keep=('messages', 20),
            # ),
            # add a todo list, which may improve agent planning capabilities
            TodoListMiddleware(),
        ],
        context_schema=CaesarRuntimeContext
    )

    # print(planner_agent.get_graph().draw_mermaid())
    # planner_agent.get_graph().draw_mermaid_png(output_file_path='agent_state_machine_graph.png')

    # code llm, this generates kernels
    code_llm = get_llm_client(**base_llm_opts)

    # prompt llm, this formats the input prompt for the code generator model
    prompt_llm = get_llm_client(**base_llm_opts)

    return planner_agent, code_llm, prompt_llm


def init_and_run_graph(
    config: CaesarRunConfig,
    work: WorkArgs,
    process_id: int,
    orchestrator: GPUOrchestrator,
    progress: mp.Value,
    worker_semaphore: mp.Semaphore,
):
    planner_agent, code_llm, prompt_llm = get_agents(config)

    # initialize state setup
    initial_context: CaesarRuntimeContext = {
        'process_id': process_id,
        'config': config,
        'work': work,
        'logger': CaesarLogger(
            os.path.join(
                config.log_dir_prefix,
                config.run_group,
                config.run_name,
                work.get_log_path(),
            ),
        ),
        'ref_problem_src': read_file(work.problem_path),
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
    }

    result = planner_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"The reference PyTorch problem to generate a CUDA kernel for is:\n\n{initial_context['ref_problem_src']}",
        }]},
        context=initial_context,
    )

    # print("---------------", result['messages'][-1].content)

    import sys
    sys.exit(0)


def run_state_machine(
    process_id: int,
    config: CaesarRunConfig,
    workargs: WorkArgs,
    orchestrator: GPUOrchestrator,
    progress: mp.Value,
    worker_semaphore: mp.Semaphore,
):
    init_and_run_graph(config, # need
                       workargs, # TODO just problem id
                       process_id, # need for printing stuff
                       orchestrator, # need for gpu
                       progress, # TODO don't really need
                       worker_semaphore) # need for compiling
