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


## system prompts for llms
COMPILE_SUMMARY_SYSTEM_PROMPT = """You are a judge, tasked with summarizing information for a failed compilation of a CUDA kernel, in a way that allows a kernel writer to fix any issues.

You will receive a compiler trace from the command line for a failed compilation of a CUDA kernel. You need to summarize this information (in a few paragraphs max) in a way that exposes the most important parts (such as the error itself, where it happened in the source file or source code, what it means exactly etc.), ignores irrelevant bits in the output (such as file paths on the current system, other function stack frames in libraries outside the user code etc.), and includes any important information from the compiler trace as-is if you consider that to be important.

This information will be used by an expert CUDA writer to fix errors. Be specific and detailed, and offer ACTIONABLE, CONCRETE suggestions for fixes and improvements.

DO NOT:
- invent anything (such as non-existent errors)
- give any other feedback other than what is in the compiler trace

DO:
- include all the important information from the trace regarding the code
- include relevant bits from the compiler trace as-is if directly relevant to fixing the code (i.e. the last stack frame + error)

Don't output anything else other than the points mentioned above!\n"""

COMPILE_SUMMARY_USER_INPUT = """Generated CUDA kernel code:

```python
{kernel_code}
```

Compiler standard output:
{stdout}

Compiler standard error:
{stderr}\n"""

RUNTIME_SUMMARY_SYSTEM_PROMPT = """You are a judge, tasked with summarizing information regarding CUDA kernel runtime, in order for an expert kernel writer to fix any issues.

You will receive a runtime trace from the command line for a failed runtime execution of an LLM-generated CUDA kernel, tested on some inputs against its reference PyTorch implementation. You need to summarize this information in a way that exposes the most important parts, ignores irrelevant bits in the output (such as file paths on the current system, other function stack frames in libraries outside the user code etc.), and includes any important information from the runtime trace as-is if you consider that to be important.

You need to present this information in ACTIONABLE steps and concrete improvement suggestions, such that a kernel writer can use it to further improve a kernel.

DO NOT:
- invent anything (such as non-existent errors)
- give any other feedback other than what is in the trace

DO:
- include all the important information from the trace regarding the code
- include relevant bits from the trace as-is if directly relevant to fixing the code
- adhere to the example given for including a CUDA kernel in a python source file; don't allow the use of `extern "C"` or other such extraneous constructions

Don't output any other text aside from the points mentioned above!\n"""

RUNTIME_SUMMARY_USER_INPUT = """Generated CUDA kernel code:

```python
{kernel_code}
```

Runtime information: {metadata}\n"""

PROFILER_SUMMARY_SYSTEM_PROMPT = """You are a judge, tasked with summarizing information regarding the performance of a CUDA kernel, in order for an expert kernel writer to further optimize the kernel.

You will receive a profiler trace for a CUDA kernel. You need to think carefully and summarize this information in a way that exposes the most important parts and ignores irrelevant bits in the output, such that a CUDA kernel writing expert can use this information to further optimize the code. Keep the text short and to the point, be brief. Ignore any unnecessary output (such as CPU code, kernel launches etc., things that cannot be influenced) which is irrelevant to the CUDA code under test. Focus just on the CUDA GPU code that can be improved in the kernel.

Your task is to summarize the main bottlenecks in the code, and come up with ACTIONABLE tasks for the kernel writer to implement, in order to improve the kernel performance.

DO NOT:
- invent anything (such as non-existent information)
- give any other feedback other than what is in the trace
- be too verbose with the text, DO include all the important information from the trace regarding hotspots

Don't output any other text aside from the mentioned points above!\n"""

PROFILER_SUMMARY_USER_INPUT = """Generated CUDA kernel code:

```python
{kernel_code}
```

Profiler trace:

{profiler_output}\n"""
