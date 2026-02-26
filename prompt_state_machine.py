import os
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, START
from langgraph.runtime import Runtime
from langgraph.graph.state import CompiledStateGraph, StateGraph

from KernelBenchInternal.utils import read_file

from caesar_config import CaesarRunConfig

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
    REFLECTION_PROFILER_FEEDBACK_INSTRUCTION,
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


@dataclass
class PromptKernelContext:
    kernel_code: dict[int, str] = field(default_factory=dict)
    eval_result: dict[int, object] = field(default_factory=dict)
    compile_summary: dict[int, dict] = field(default_factory=dict)
    runtime_summary: dict[int, dict] = field(default_factory=dict)
    profiler_summary: dict[int, dict] = field(default_factory=dict)


@dataclass
class PromptRuntimeContext:
    config: CaesarRunConfig
    turn: int
    kernel_context: PromptKernelContext


class PromptGraphState(TypedDict):
    prompt: str
    best_kernel_idx: int | None
    last_kernel_idx: int | None


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


def empty_prompt_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    return { 'prompt': '' }


def best_and_last_kernel_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.kernel_context.eval_result
    kernels = runtime.context.kernel_context.kernel_code
    prompt = state['prompt']

    best_kernel_idx: int | None = get_best_kernel_code(eval_result)
    last_kernel_idx: int | None = get_last_kernel_code(kernels)

    if best_kernel_idx is None or best_kernel_idx == last_kernel_idx:
        prompt += PREVIOUSLY_GENERATED_KERNEL.format(
            prev_kernel_code=kernels[last_kernel_idx]
        )
    elif best_kernel_idx is not None and best_kernel_idx != last_kernel_idx:
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
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> Literal[
    'compiler_feedback_handler',
    'correctness_feedback_handler',
    'profiler_feedback_handler',
    'final_prompt_handler'
]:
    config = runtime.context.config
    eval_result = runtime.context.kernel_context.eval_result
    last_kernel_idx = state['last_kernel_idx']

    if (
        config.use_compiler_feedback
        and eval_result[last_kernel_idx].metadata != {}
        and eval_result[last_kernel_idx].compiled is False
    ):
        return 'compiler_feedback_handler'

    if (
        config.use_correctness_feedback
        and eval_result[last_kernel_idx].metadata != {}
        and eval_result[last_kernel_idx].compiled is True
        and eval_result[last_kernel_idx].correctness is False
    ):
        return 'correctness_feedback_handler'

    if config.use_profiler_feedback:
        return 'profiler_feedback_handler'

    return 'final_prompt_handler'


def compiler_feedback_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.kernel_context.eval_result
    compile_summary = runtime.context.kernel_context.compile_summary
    prompt = state['prompt']
    last_kernel_idx = state['last_kernel_idx']

    if (
        eval_result[last_kernel_idx].metadata != {}
        and eval_result[last_kernel_idx].compiled is False
    ):
        prompt += COMPILER_FEEDBACK_PROMPT.format(
            compiler_feedback=compile_summary[last_kernel_idx]["content"]
        )
        prompt += REFLECTION_COMPILER_FEEDBACK_INSTRUCTION

    return { 'prompt': prompt }


def correctness_feedback_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    runtime_summary = runtime.context.kernel_context.runtime_summary
    prompt = state['prompt']
    last_kernel_idx = state['last_kernel_idx']

    prompt += CORRECTNESS_FEEDBACK_PROMPT.format(
        correctness_feedback=runtime_summary[last_kernel_idx]['content'],
    )
    prompt += REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION
    return { 'prompt': prompt }


def profiler_feedback_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.kernel_context.eval_result
    profiler_summary = runtime.context.kernel_context.profiler_summary
    prompt = state['prompt']
    best_kernel_idx = state['best_kernel_idx']
    last_kernel_idx = state['last_kernel_idx']

    if (
        best_kernel_idx is not None
        and profiler_summary.get(best_kernel_idx, "") != ""
    ):
        prompt += PROFILER_FEEDBACK_PROMPT.format(
            kernel="best",
            profiler_feedback=profiler_summary[best_kernel_idx]['content'],
            runtime_ms=eval_result[best_kernel_idx].runtime,
        )

    if (
        last_kernel_idx != best_kernel_idx
        and profiler_summary.get(last_kernel_idx, "") != ""
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
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    prompt = state['prompt']
    prompt += REFLECTION_INSTRUCTION
    return { 'prompt': prompt }


def _init_prompt_state_machine_graph() -> CompiledStateGraph:
    builder = StateGraph(PromptGraphState, context_schema=PromptRuntimeContext)

    builder.add_node('empty_prompt_handler', empty_prompt_handler)
    builder.add_node('best_and_last_kernel_handler', best_and_last_kernel_handler)
    builder.add_node('compiler_feedback_handler', compiler_feedback_handler)
    builder.add_node('correctness_feedback_handler', correctness_feedback_handler)
    builder.add_node('profiler_feedback_handler', profiler_feedback_handler)
    builder.add_node('final_prompt_handler', final_prompt_handler)

    builder.add_conditional_edges(
        START,
        lambda state, runtime:
            'empty_prompt_handler'
            if (
                runtime.context.kernel_context.kernel_code is None
                or all(
                    not v for v in runtime.context.kernel_context.kernel_code.values()
                )
            ) else
            'best_and_last_kernel_handler',
        ['empty_prompt_handler', 'best_and_last_kernel_handler']
    )
    builder.add_conditional_edges('best_and_last_kernel_handler',
                                  feedback_decision)
    builder.add_edge('empty_prompt_handler', END)
    builder.add_edge('compiler_feedback_handler', END)
    builder.add_edge('correctness_feedback_handler', END)
    builder.add_edge('profiler_feedback_handler', END)
    builder.add_edge('final_prompt_handler', END)

    return builder.compile()


def build_llm_prompt(
    config: CaesarRunConfig,
    turn: int,
    kernel_context: PromptKernelContext,
) -> str:
    prompt_graph = _init_prompt_state_machine_graph()

    prompt = prompt_graph.invoke({
        'prompt': '',
        'best_kernel_idx': None,
        'last_kernel_idx': None,
    }, context={
        'config': config,
        'turn': turn,
        'kernel_context': kernel_context,
    })['prompt']

    return prompt

