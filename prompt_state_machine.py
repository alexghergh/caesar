import os
from typing import TypedDict
from dataclasses import dataclass

from KernelBenchInternal.utils import read_file
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from caesar_config import CaesarRunConfig
from conversation_info import ConversationInfo
from rag import RagIndex
from prompts import (
    CODE_AGENT_SYSTEM_PROMPT,
    PROMPT_AGENT_USER_INPUT,
)


REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench", "KernelBench")
KERNEL_BENCH_ARCH_EXAMPLES_PATH = os.path.join(
    REPO_TOP_PATH, "KernelBench", "KernelBenchInternal", "prompts"
)


# fed to the prompting agent
@dataclass
class RefinedPromptContext:
    rag_index: RagIndex
    conv_info: ConversationInfo
    rag_scope: str
    rag_top_k: int
    problem_id: int
    turn: int


def build_code_agent_system_prompt(
    config: CaesarRunConfig,
    ref_arch_src: str,
) -> str:
    """
    Build the code agent system prompt with hardware, examples, and the
    reference architecture inlined.
    """
    example_ind = 'add'
    example_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_new_ex_{example_ind}.py"
    )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return CODE_AGENT_SYSTEM_PROMPT.format(
        hardware_list=", ".join(config.gpu_arch),
        example_arch_src=example_arch,
        example_new_arch_src=example_new_arch,
        arch_src=ref_arch_src,
    )


def build_llm_prompt(
    config: CaesarRunConfig,
    prompt_agent: CompiledStateGraph,
    turn: int,
    problem_id: int,
    thread_id: str,
    rag_index: RagIndex,
    conv_info: ConversationInfo,
) -> str:

    if turn == 1:
        # initial prompt, no info
        base_prompt = "No kernel generated so far, as it is the first turn. The coding agent already has access to the problem's specifications, so just ask it to generate a kernel for now."
    else:
        eval_result = conv_info.eval_result
        kernels = conv_info.kernel_code

        # runtime of last kernel (or fail if no kernel generated)
        kernel = kernels.get(turn - 1, 'failed to generate kernel')
        runtime_perf = eval_result.get(turn - 1, 'failed to generate kernel').runtime

        # get the feedback from last turn
        if (feedback := conv_info.compile_summary.get(turn - 1, '')) != '':
            phase = 'compilation'
            feedback = feedback['content']
        elif (feedback := conv_info.runtime_summary.get(turn - 1, '')) != '':
            phase = 'runtime correctness'
            feedback = feedback['content']
        elif (feedback := conv_info.profiler_summary.get(turn - 1, '')) != '':
            phase = 'profiling'
            feedback = feedback['content']
        else:
            phase = 'unknown (no information given)'
            feedback = 'no feedback'

        # build the base prompt; this should just contain the necessary
        # information + human prompt from last turns (e.g. feedback + generated
        # best/last kernels)
        base_prompt = PROMPT_AGENT_USER_INPUT.format(
            turn=turn - 1,
            runtime_ms=runtime_perf,
            kernel_code=kernel,
            phase=phase,
            feedback=feedback,
        )

    # base prompt
    conv_info.prompt[turn] = base_prompt

    # invoke a prompting agent to refine this prompt
    response = prompt_agent.invoke({
        "messages": [{
            "role": "user",
            "content": base_prompt,
        }],
        "configurable": {
            "thread_id": thread_id,
        }
    }, context=RefinedPromptContext(
        rag_index=rag_index,
        conv_info=conv_info,
        rag_scope=config.rag_scope,
        rag_top_k=config.rag_top_k,
        problem_id=problem_id,
        turn=turn,
    ))
    last_message: AIMessage = response["messages"][-1]
    final_prompt = last_message.text
    conv_info.formatted_prompt[turn] = final_prompt

    return final_prompt
