import os
from typing import Literal, TypedDict
from dataclasses import dataclass

from langgraph.graph import END, START
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from KernelBenchInternal.utils import read_file
from langgraph.graph.state import CompiledStateGraph, StateGraph

from caesar_config import CaesarRunConfig
from conversation_info import ConversationInfo
from rag import RagIndex
from prompts import (
    CODE_AGENT_SYSTEM_PROMPT,
    COMPILER_FEEDBACK_PROMPT,
    CORRECTNESS_FEEDBACK_PROMPT,
    PREVIOUSLY_GENERATED_BEST_AND_LAST_KERNELS,
    PREVIOUSLY_GENERATED_KERNEL,
    PROFILER_FEEDBACK_PROMPT,
    REFLECTION_COMPILER_FEEDBACK_INSTRUCTION,
    REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION,
    REFLECTION_INSTRUCTION,
    REFLECTION_PROFILER_FEEDBACK_INSTRUCTION
)
from utils import get_best_kernel_code, get_last_kernel_code


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


# these don't change across prompt setup run
@dataclass
class BasePromptCreationRuntimeContext:
    config: CaesarRunConfig
    turn: int
    ref_arch_src: str
    conv_info: ConversationInfo

# change across the prompt handler
class PromptGraphState(TypedDict):
    prompt: str
    best_kernel_idx: int | None
    last_kernel_idx: int | None


# separate data, fed to the prompting agent
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


def initial_prompt_handler(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> PromptGraphState:
    """
    Return an initial message.
    """
    return { 'prompt': 'Please generate the kernel per the specifications.' }


def best_and_last_kernel_handler(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> PromptGraphState:
    """
    Append best and last kernels to the prompt.
    """
    eval_result = runtime.context.conv_info.eval_result
    kernels = runtime.context.conv_info.kernel_code
    prompt = state['prompt']

    # get the best kernel so far in terms of runtime (we have kernel code;
    # however, it doesn't mean it compiled or ran correctly! that's why this can
    # return None) and the last generated kernel (regardless of whether it
    # compiled or ran correctly or not)
    best_kernel_idx: int | None = get_best_kernel_code(eval_result)
    last_kernel_idx: int | None = get_last_kernel_code(kernels)

    # at this point, last_kernel_idx is guaranteed non-None
    # there's a few cases to consider:
    # - best_kernel_idx is None (because no kernel compiled so far)
    # - best_kernel_idx is the same as last_kernel_idx (because the last kernel
    # compiled and ran correctly)
    # - best_kernel_idx is different from last_kernel_idx (because we have a
    # kernel that compiled and ran correctly at some previous iteration, but the
    # last generated one either didn't compile, didn't run successfully, or was
    # slower)

    if best_kernel_idx is None or best_kernel_idx == last_kernel_idx:
        # we don't have a best kernel yet OR it is the same as the
        # last kernel
        prompt += PREVIOUSLY_GENERATED_KERNEL.format(
            prev_kernel_code=kernels[last_kernel_idx]
        )
    elif best_kernel_idx is not None and best_kernel_idx != last_kernel_idx:
        # different kernels generated at different times
        prompt += PREVIOUSLY_GENERATED_BEST_AND_LAST_KERNELS.format(
            best_kernel_code=kernels[best_kernel_idx],
            last_kernel_code=kernels[last_kernel_idx],
        )

    return {
        'prompt': prompt,
        'best_kernel_idx': best_kernel_idx,
        'last_kernel_idx': last_kernel_idx
    }


def feedback_decision(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> Literal[
    'compiler_feedback_handler',
    'correctness_feedback_handler',
    'profiler_feedback_handler',
    'final_prompt_handler'
]:
    config = runtime.context.config
    eval_result = runtime.context.conv_info.eval_result
    last_kernel_idx = state['last_kernel_idx']

    # we can either give the LLM feedback for the best kernel, or for
    # all the kernels generated; as the prompts can get quite large
    # (thousands of tokens) with multiple generated kernels, we should
    # only offer the best feedback available at hand, which is either:
    # - (best case) feedback for the kernel that compiled, ran, and has
    # profiler output
    # - (worst case) feedback for the last kernel that didn't compile or
    # didn't run correctly
    # - (in-between) if there's a valid (i.e. compiler + runtime
    # correct) kernel at some previous iteration, but the kernel at the
    # current iteration didn't compile or run correctly, or was slower
    # than the best kernel, than tell the model about both
    #
    # as an action plan, we always offer compiler, correctness and
    # profiler feedback for the last kernel, and profiler feedback for
    # the best kernel when the last kernel is slower
    if (
        config.use_compiler_feedback
        and eval_result[last_kernel_idx].metadata != {} # always True
        and eval_result[last_kernel_idx].compiled is False # always True
    ):
        return 'compiler_feedback_handler'

    # this assumes that correctness check is empty if compile failed
    # offer correctness check feedback if it failed; otherwise, move
    # on to profiler feedback
    if (
        config.use_correctness_feedback
        and eval_result[last_kernel_idx].metadata != {}
        and eval_result[last_kernel_idx].compiled is True
        and eval_result[last_kernel_idx].correctness is False
    ):
        return 'correctness_feedback_handler'

    # best is none, last runtime issue
    # best != last
    # best == last
    if config.use_profiler_feedback:
        return 'profiler_feedback_handler'

    return 'final_prompt_handler'


def compiler_feedback_handler(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.conv_info.eval_result
    compile_summary = runtime.context.conv_info.compile_summary
    prompt = state['prompt']
    last_kernel_idx = state['last_kernel_idx']

    # offer compiler feedback if compilation failed; otherwise, move
    # on to correctness check
    if (
        eval_result[last_kernel_idx].metadata != {} # always True
        and eval_result[last_kernel_idx].compiled is False # always True
    ):
        prompt += COMPILER_FEEDBACK_PROMPT.format(
            compiler_feedback=compile_summary[last_kernel_idx]["content"]
        )
        prompt += REFLECTION_COMPILER_FEEDBACK_INSTRUCTION

    return { 'prompt': prompt }


def correctness_feedback_handler(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> PromptGraphState:
    runtime_summary = runtime.context.conv_info.runtime_summary
    prompt = state['prompt']
    last_kernel_idx = state['last_kernel_idx']

    prompt += CORRECTNESS_FEEDBACK_PROMPT.format(
        correctness_feedback=runtime_summary[last_kernel_idx]['content'],
    )
    prompt += REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION
    return { 'prompt': prompt }


def profiler_feedback_handler(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.conv_info.eval_result
    profiler_summary = runtime.context.conv_info.profiler_summary
    prompt = state['prompt']
    best_kernel_idx = state['best_kernel_idx']
    last_kernel_idx = state['last_kernel_idx']

    # always include best kernel profiler feedback if available
    if (
        best_kernel_idx is not None
        and profiler_summary.get(best_kernel_idx, "") != ""
    ):
        prompt += PROFILER_FEEDBACK_PROMPT.format(
            kernel="best",
            profiler_feedback=profiler_summary[best_kernel_idx]['content'],
            runtime_ms=eval_result[best_kernel_idx].runtime,
        )

    # include last kernel profiler feedback if it was slower;
    # if it was faster, then by definition the last kernel IS the
    # best kernel
    if (
        last_kernel_idx != best_kernel_idx

        # if there's no profiler feedback, we can be sure something
        # was wrong during compilation or runtime; skip the rest of
        # the checks
        and profiler_summary.get(last_kernel_idx, "") != ""

        # last kernel is slower than the best kernel
        and eval_result[last_kernel_idx].runtime >
            eval_result[best_kernel_idx].runtime
    ):
        prompt += PROFILER_FEEDBACK_PROMPT.format(
            kernel="previous",
            profiler_feedback=profiler_summary[last_kernel_idx]['content'],
            runtime_ms=eval_result[last_kernel_idx].runtime,
        )

    prompt += REFLECTION_PROFILER_FEEDBACK_INSTRUCTION
    return { 'prompt': prompt }


def final_prompt_handler(
    state: PromptGraphState, runtime: Runtime[BasePromptCreationRuntimeContext]
) -> PromptGraphState:
    prompt = state['prompt']
    prompt += REFLECTION_INSTRUCTION
    return { 'prompt': prompt }


def _init_prompt_state_machine_graph() -> CompiledStateGraph:
    builder = StateGraph(
        PromptGraphState, context_schema=BasePromptCreationRuntimeContext
    )

    # prompt states
    builder.add_node('initial_prompt_handler', initial_prompt_handler)
    builder.add_node('best_and_last_kernel_handler', best_and_last_kernel_handler)
    builder.add_node('compiler_feedback_handler', compiler_feedback_handler)
    builder.add_node('correctness_feedback_handler', correctness_feedback_handler)
    builder.add_node('profiler_feedback_handler', profiler_feedback_handler)
    builder.add_node('final_prompt_handler', final_prompt_handler)

    # transitions
    builder.add_conditional_edges(
        START,
        lambda state, runtime:
            'initial_prompt_handler'
            if (
                runtime.context.turn == 1
                or runtime.context.conv_info.kernel_code is None
                or all(not v for v in runtime.context.conv_info.kernel_code.values())
            ) else
            'best_and_last_kernel_handler',
        ['initial_prompt_handler', 'best_and_last_kernel_handler']
    )
    builder.add_conditional_edges('best_and_last_kernel_handler', feedback_decision)
    builder.add_edge('initial_prompt_handler', END)
    builder.add_edge('compiler_feedback_handler', END)
    builder.add_edge('correctness_feedback_handler', END)
    builder.add_edge('profiler_feedback_handler', END)
    builder.add_edge('final_prompt_handler', END)

    graph = builder.compile()
    # print(graph.get_graph().draw_mermaid())
    # graph.get_graph().draw_mermaid_png(output_file_path='prompt_sm_graph.png')
    return graph


def build_llm_prompt(
    config: CaesarRunConfig,
    prompt_agent: CompiledStateGraph,
    turn: int,
    problem_id: int,
    ref_arch_src: str,
    rag_index: RagIndex,
    conv_info: ConversationInfo,
) -> str:

    # init prompt builder graph
    prompt_graph = _init_prompt_state_machine_graph()

    # build the base prompt; this should just contain the necessary information
    # + human prompt from last turns (e.g. feedback + generated best/last kernels)
    base_prompt = prompt_graph.invoke({
        'prompt': '',
        'best_kernel_idx': None,
        'last_kernel_idx': None,
    }, context={
        'config': config,
        'turn': turn,
        'ref_arch_src': ref_arch_src,
        'conv_info': conv_info,
    })['prompt']
    conv_info.prompt[turn] = base_prompt

    if base_prompt.strip() == "":
        conv_info.formatted_prompt[turn] = ""
        return ""

    # invoke a prompting agent to refine this prompt
    response = prompt_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": base_prompt,
                }
            ]
        },
        context=RefinedPromptContext(
            rag_index=rag_index,
            conv_info=conv_info,
            rag_scope=config.rag_scope,
            rag_top_k=config.rag_top_k,
            problem_id=problem_id,
            turn=turn,
        ),
    )
    last_message: AIMessage = response["messages"][-1]
    final_prompt = last_message.content
    conv_info.formatted_prompt[turn] = final_prompt

    return final_prompt
