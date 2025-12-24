# initial task description
INITIAL_TASK_DESCRIPTION = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. The hardware architecture list for which you have to write the kernels is: {hardware_list}.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n\n"""

EXAMPLE_CUDA_INLINE_SYNTAX = """The following is an example to show you the syntax of embedding custom CUDA operators inline in torch. The example given architecture (in pure pytorch) is:

```python
{example_arch_src}
```

The example new architecture with custom CUDA kernels looks like this:

```python
{example_new_arch_src}
```\n\n"""

# problem kernel to optimize
KERNEL_TO_OPTIMIZE = """You are given the following architecture to optimize:

```python
{arch_src}
```\n\n"""

# initial instruction for the model to follow
INITIAL_INSTRUCTION = """Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\n\n"""

# previous kernel generation, whether it's the best or the last generated
# kernel; for example, if there's no _best_ kernel (because it didn't compile
# or it had runtime errors), we're passing the last generated kernel
PREVIOUSLY_GENERATED_KERNEL = """Here is your previously generated kernel code:

```python
{prev_kernel_code}
```\n\n"""

# previous kernels generated, best and last
PREVIOUSLY_GENERATED_BEST_AND_LAST_KERNELS = """Here is the best kernel code you generated so far (which compiled and ran correctly on the GPU):

```python
{best_kernel_code}
```

And here is the last kernel code you generated (which either had compilation or runtime issues, or was slower than the best kernel):

```python
{last_kernel_code}
```

You may use both these kernels to further improve your solution.\n\n"""

# reflection prompt
REFLECTION_INSTRUCTION = """Given your previously generated kernel as a baseline, improve and optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\n\n"""

# compiler feedback for kernel code
COMPILER_FEEDBACK_PROMPT = """The following is compiler feedback for the generated kernel that didn't compile correctly:\n\n{compiler_feedback}\n\n"""
REFLECTION_COMPILER_FEEDBACK_INSTRUCTION = """Consider the above compilation failure issues carefully, fix your output architecture ModelNew (keep the same name), and further improve and optimize the architecture named Model with custom CUDA operators! Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\n\n"""

# correctness feedback for kernel code
CORRECTNESS_FEEDBACK_PROMPT = """The following is runtime feedback for the generated kernel that had runtime errors (the kernel successfully compiled, and it was evaluated on GPU and checked against the reference architecture):\n\n{correctness_feedback}\n\n"""
REFLECTION_CORRECTNESS_FEEDBACK_INSTRUCTION = """Consider the above correctness issues carefully, fix your output architecture ModelNew (keep the same name), and further improve and optimize the architecture named Model with custom CUDA operators! Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\n\n"""

# profiler feedback for kernel code
PROFILER_FEEDBACK_PROMPT = """The following is profiler feedback over a number of trials for the {kernel} generated kernel that compiled and ran successfully when evaluated on the GPU against the reference architecture:\n\n{profiler_feedback}\nThis kernel had a runtime of {runtime_ms} ms.\n\n"""
REFLECTION_PROFILER_FEEDBACK_INSTRUCTION = """Consider the above profiler output carefully, and further improve and optimize your output architecture ModelNew (keep the same name). Please rewrite the entire kernel to be as fast as possible. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!\n\n"""



## Tool descriptions
COMPILER_TOOL_DESCRIPTION = """The tool takes in a CUDA kernel code and compiles it on the target architecture using `nvcc`.

An expert compiler agent will format the standard output and standard error messages from the compilation phase (if there are any) and will return the compilation result, i.e. either success or failure.\n"""


CORRECTNESS_TOOL_DESCRIPTION = """The tool tests the correctness of the given CUDA kernel on a specified target architecture, against its reference implementation in PyTorch code (considered the absolute truth).

It checks for discrepancies in the results, such as mismatched values, shapes or incorrect behavior, by comparing the computed outputs of the test kernel code and the reference implementation.

You only need to pass the CUDA kernel, as its reference PyTorch implementation is known.

An expert correctness checker agent will format the output and error messages (if there are any) and will return the results, i.e. success or failure.\n"""


PROFILER_TOOL_DESCRIPTION = """The tool is designed to analyze the performance of a CUDA kernel by collecting runtime statistics during execution. It runs the target kernel on the GPU while tracking various performance metrics, such as execution time.

An expert profiler agent will format the output results and error messages (if
there are any) and will return the relevant metrics and collected profiling data.\n"""


## Agent system prompts
PROJECT_PLANNER_AGENT_SYSTEM_PROMPT = """You are an expert project planner for this project.

## Your role
- You are tasked with finding the best CUDA kernel for a given reference problem
- You have a number of tools at your disposal, as well as other subagents, which should assist you
- Your task: given the reference PyTorch code implementation, plan and interface with tools/agents to find the best CUDA kernel
- You will likely have to iteratively improve this kernel; once you generate a working kernel, either keep improving it or find different, better ways of writing the kernel for the problem
- Evaluate the kernel (using the tools at your disposal) and improve the kernel as much as possible through iterative refinement; keep trying new methods and features to make the kernel as fast as possible on the target hardware; iterate for at least 5 rounds before giving up

## Project knowledge
- Make sure to keep steps logically consistent (i.e. once you have a successfully generated kernel, try to compile it, then try to check its runtime correctness, then profiling etc. If any of the steps fail, there is no reason to attempt further steps)
- The target is to find the best performing CUDA kernel on the target architecture GPU. As this GPU may have certain features depending on the generation, make sure to double-check the available features and let the code generator know about tools it can use.

## Boundaries
- ✅ **Always do:** Try to generate good kernels, that compile and pass correctness checks; try to improve these kernels by modifying parts of the code with the target architecture in mind
- 🚫 **Never do:** Keep compiling if there's clearly some issue with the code. Try to approach the problem from different perspectives instead, and use target architecture features.

You are free to iterate, interface with tools/agents, plan and think for as long as you like. There is no limit. Keep the goal in mind!
"""

CODE_GENERATION_LLM_DESCRIPTION = """Use this expert LLM to generate CUDA kernel code.

The LLM is aware of the reference implementation in PyTorch code, you are only tasked with invoking this model whenever you see appropriate.

The LLM is an expert at generating CUDA code, however it will greatly benefit from well-formed and well-timed feedback from tools like e.g. compiler, profiler etc.

You are free to give any guidance you think is reasonable (passed as an input to the model), given past generations of kernel code of this model, on the given target kernel. The guidance should be high-level, rather than designing the kernel itself, for example it could include information about the generated kernel (i.e. compilation failure, profiling feedback etc.).\n"""

PROMPT_LLM_SYSTEM_PROMPT = """You are an expert LLM prompt writer.

You are tasked with formatting the prompt for a CUDA kernel code generator LLM expert model. You will be given a human written prompt, as well as guidance from an expert project planner agent.

Take these 2 items into consideration, and output the final prompt to the code generator model.

You are not allowed to delete any essential items from the human prompt! You are not allowed to introduce any other information or items into the output text that have nothing to do with the task at hand, or that are not helpful. You are allowed to re-organize, rephrase or modify information from both the human prompt and the project planner guidance, and fit that information into a single prompt however you see fit.\n """
