# TODOs:
# - CoT (i.e. reasoning over how to optimize a kernel)
# - few-shot examples
# - summaries into new rounds (i.e. prompt an LLM to summarize the current
# round kernel and optimizations applied, for the next round)
# - GePA
# - RAG few-shot or related optimizations
# - hardware information
# - other somewhat automated optimizations (i.e. performance measurements over
# CUDA kernel block size)



# flow explanation

# round 1
#   {INITIAL_TASK_DESCRIPTION}
#   optional {EXAMPLE_CUDA_INLINE_SYNTAX}
#   {KERNEL_TO_OPTIMIZE}
#   {INITIAL_INSTRUCTION}


# round n (if kernel generation succeeded):
#   with no compile/correctness check/profiler feedback
#       {INITIAL_TASK_DESCRIPTION}
#       optional {EXAMPLE_CUDA_INLINE_SYNTAX}
#       {KERNEL_TO_OPTIMIZE}
#       {PREVIOUSLY_GENERATED_BEST_KERNEL}
#       {REFLECTION_INSTRUCTION}
# TODO

# initial task description
INITIAL_TASK_DESCRIPTION = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n\n"""

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
