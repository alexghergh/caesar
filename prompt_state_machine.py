import os
from typing import Literal, TypedDict
from dataclasses import dataclass

from langgraph.graph import END, START
from langgraph.runtime import Runtime

from KernelBenchInternal.utils import read_file
from langgraph.graph.state import CompiledStateGraph, StateGraph

from caesar_config import CaesarRunConfig
from prompts import (
    COMPILER_FEEDBACK_PROMPT,
    CORRECTNESS_FEEDBACK_PROMPT,
    EXAMPLE_CUDA_INLINE_SYNTAX,
    INITIAL_TASK_DESCRIPTION,
    INITIAL_INSTRUCTION,
    KERNEL_TO_OPTIMIZE,
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


@dataclass
class PromptRuntimeContext:
    config: CaesarRunConfig
    turn: int
    ref_arch_src: str
    kernels: dict
    eval_result: dict
    profiler_result: dict
    max_profiler_feedback_length: int


class PromptGraphState(TypedDict):
    prompt: str
    best_kernel_idx: int | None
    last_kernel_idx: int | None


def problem_statement_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    """
    Construct an initial template prompt to show to the model.
    Additionally, it contains an example implementation of a custom CUDA kernel
    in PyTorch.
    """
    # initial prompt (always start from the task description + reference kernel
    # to optimize)

    # example kernel to show syntax (addition kernel)
    example_ind = 'add'
    example_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_new_ex_{example_ind}.py"
    )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    # construct the initial prompt
    prompt = INITIAL_TASK_DESCRIPTION.format(
        hardware_list=", ".join(runtime.context.config.gpu_arch)
    )

    prompt += EXAMPLE_CUDA_INLINE_SYNTAX.format(
        example_arch_src=example_arch, example_new_arch_src=example_new_arch
    )

    prompt += KERNEL_TO_OPTIMIZE.format(arch_src=runtime.context.ref_arch_src)

    return { 'prompt': prompt }


def initial_instruction_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    """
    Add initial instruction to the prompt.
    """
    prompt = state['prompt']
    prompt += INITIAL_INSTRUCTION
    return { 'prompt': prompt }


def best_and_last_kernel_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    """
    Append best and last kernels to the prompt.
    """
    eval_result = runtime.context.eval_result
    kernels = runtime.context.kernels
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
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> Literal[
    'compiler_feedback_handler',
    'correctness_feedback_handler',
    'profiler_feedback_handler',
    'final_prompt_handler'
]:
    config = runtime.context.config
    eval_result = runtime.context.eval_result
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
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.eval_result
    prompt = state['prompt']
    last_kernel_idx = state['last_kernel_idx']

    # offer compiler feedback if compilation failed; otherwise, move
    # on to correctness check
    if (
        eval_result[last_kernel_idx].metadata != {} # always True
        and eval_result[last_kernel_idx].compiled is False # always True
    ):
        metadata = eval_result[last_kernel_idx].metadata
        metadata.pop("hardware", None)
        metadata.pop("device", None)
        key = next(iter(metadata))

        prompt += COMPILER_FEEDBACK_PROMPT.format(
            compiler_feedback=f"{key}: {metadata[key]}"
        )
        prompt += REFLECTION_COMPILER_FEEDBACK_INSTRUCTION

    return { 'prompt': prompt }


def correctness_feedback_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.eval_result
    prompt = state['prompt']
    last_kernel_idx = state['last_kernel_idx']

    metadata = eval_result[last_kernel_idx].metadata
    metadata.pop("hardware", None)
    metadata.pop("device", None)
    issue = metadata.get("correctness_issue", "")
    issue = metadata.get("runtime_error", "") if issue == "" else issue

    prompt += CORRECTNESS_FEEDBACK_PROMPT.format(
        correctness_feedback=f"{issue}"
    )
    prompt += REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION
    return { 'prompt': prompt }


def profiler_feedback_handler(
    state: PromptGraphState, runtime: Runtime[PromptRuntimeContext]
) -> PromptGraphState:
    eval_result = runtime.context.eval_result
    profiler_result = runtime.context.profiler_result
    max_profiler_feedback_length = runtime.context.max_profiler_feedback_length
    prompt = state['prompt']
    best_kernel_idx = state['best_kernel_idx']
    last_kernel_idx = state['last_kernel_idx']

    # always include best kernel profiler feedback if available
    if (
        best_kernel_idx is not None
        and profiler_result.get(best_kernel_idx, "") != ""
    ):
        prompt += PROFILER_FEEDBACK_PROMPT.format(
            kernel="best",
            profiler_feedback=profiler_result[best_kernel_idx][
                :max_profiler_feedback_length
            ],
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
        and profiler_result.get(last_kernel_idx, "") != ""

        # last kernel is slower than the best kernel
        and eval_result[last_kernel_idx].runtime >
            eval_result[best_kernel_idx].runtime
    ):
        prompt += PROFILER_FEEDBACK_PROMPT.format(
            kernel="previous",
            profiler_feedback=profiler_result[last_kernel_idx][
                :max_profiler_feedback_length
            ],
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

    # prompt states
    builder.add_node('problem_statement_handler', problem_statement_handler)
    builder.add_node('initial_instruction_handler', initial_instruction_handler)
    builder.add_node('best_and_last_kernel_handler', best_and_last_kernel_handler)
    builder.add_node('compiler_feedback_handler', compiler_feedback_handler)
    builder.add_node('correctness_feedback_handler', correctness_feedback_handler)
    builder.add_node('profiler_feedback_handler', profiler_feedback_handler)
    builder.add_node('final_prompt_handler', final_prompt_handler)

    # transitions
    builder.add_edge(START, 'problem_statement_handler')
    builder.add_edge('initial_instruction_handler', END)
    builder.add_conditional_edges(
        'problem_statement_handler',
        lambda state, runtime:
            'initial_instruction_handler'
            if (
                # check whether it's turn 1, or we have any kernels generated
                # so far; if we don't have a valid kernel code so far, re-prompt
                # using the initial prompt
                runtime.context.turn == 1
                or runtime.context.kernels is None
                or all(not v for v in runtime.context.kernels.values())
            ) else
            'best_and_last_kernel_handler',
        ['initial_instruction_handler', 'best_and_last_kernel_handler']
    )
    builder.add_conditional_edges('best_and_last_kernel_handler',
                                  feedback_decision)
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
    turn: int,
    ref_arch_src: str,
    kernels: dict,
    eval_result: dict,
    profiler_result: dict,
    max_profiler_feedback_length: int,
) -> PromptGraphState:

    # init prompt builder graph
    prompt_graph = _init_prompt_state_machine_graph()

    # run
    return prompt_graph.invoke({
        'prompt': '',
        'best_kernel_idx': None,
        'last_kernel_idx': None,
    }, context={
        'config': config,
        'turn': turn,
        'ref_arch_src': ref_arch_src,
        'kernels': kernels,
        'eval_result': eval_result,
        'profiler_result': profiler_result,
        'max_profiler_feedback_length': max_profiler_feedback_length
    })['prompt']
