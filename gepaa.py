import os
from typing import Any, Mapping, Sequence, override

import gepa
from gepa.adapters.default_adapter.default_adapter import ChatMessage, DefaultReflectiveRecord, DefaultRolloutOutput, DefaultTrajectory
from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter, RolloutOutput, Trajectory

from KernelBenchInternal.utils import (
    extract_last_code,
    read_file,
)
from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_SUBSET_DATASET,
)

from eval import compile_single_sample
from prompts import EXAMPLE_CUDA_INLINE_SYNTAX, INITIAL_INSTRUCTION, INITIAL_TASK_DESCRIPTION, KERNEL_TO_OPTIMIZE

trainset, valset, _ = gepa.examples.aime.init_dataset()

dataset = KernelBenchDataset(KERNELBENCH_LEVEL_1_SUBSET_DATASET)

# TODO HERE basically just import the KernelBenchDataset, and transform it into
# this trainset from below
# then, update the system_prompt
# then, build the correct input prompt to the model, and then the output prompt
# as feedback
# don't forget to mention stuff like H100 etc.
trainset = []
for problem_id in dataset.get_problem_ids():
    input_file = dataset.get_problem_path_by_id(problem_id)
    inp = read_file(input_file)
    trainset.append({ 'pytorch_kernel': inp })

#
# system prompt
#
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
example_ind = 'add'
example_arch_path = os.path.join(
    KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_ex_{example_ind}.py"
)
example_new_arch_path = os.path.join(
    KERNEL_BENCH_ARCH_EXAMPLES_PATH, f"model_new_ex_{example_ind}.py"
)
example_arch = read_file(example_arch_path)
example_new_arch = read_file(example_new_arch_path)
seed_prompt = {
    "system_prompt": INITIAL_TASK_DESCRIPTION
    + EXAMPLE_CUDA_INLINE_SYNTAX.format(
        example_arch_src=example_arch, example_new_arch_src=example_new_arch
    )
    + INITIAL_INSTRUCTION
}




class CudaAdapter(GEPAAdapter):

    def __init__(self, model):
        import litellm
        self.litellm = litellm

        self.model = model
        # self.failure_score = failure_score
        # self.max_litellm_workers = max_litellm_workers
        # self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs

    @override
    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:

        print("called evaluate", capture_traces)

        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        litellm_requests = []

        # the inputs are the system prompt, which is the "you are a cuda
        # engineer..", and the actual cuda problem to optimize
        for data in batch:
            user_content = KERNEL_TO_OPTIMIZE.format(arch_src=data['pytorch_kernel'])

            messages: list[ChatMessage] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            litellm_requests.append(messages)

        # get the llm's responses to the problems, i.e. the generated kernels
        try:
            if isinstance(self.model, str):
                responses = [
                    resp.choices[0].message.content.strip()
                    for resp in self.litellm.batch_completion(
                        model=self.model,
                        api_key='SGLANG_KEY',
                        api_base='http://localhost:34561/v1/',
                        messages=litellm_requests,
                        max_workers=3,
                    )
                ]
            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception as e:
            raise e

        # now that we have the responses, we need to compile, run correctness,
        # run profiler, etc.
        for data, assistant_response in zip(batch, responses, strict=False):
            output: DefaultRolloutOutput = {
                "full_assistant_response": assistant_response
            }

            # let's assume for now we assign a score of 1 if the kernel compiles
            kernel_code = extract_last_code(assistant_response, ["python", "cpp"])
            assert kernel_code is not None and len(kernel_code) != 0
            returncode, out, err = compile_single_sample(
                kernel_src=kernel_code,
                gpu_arch=["Hopper"],
                build_dir="test-build-dir",
                timeout_seconds=6000,
            )

            score = 1.0 if returncode == 0 else 0.0

            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "kernel_code": kernel_code,
                        "compiler_output": err,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    @override
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ret_d: dict[str, list[DefaultReflectiveRecord]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[DefaultReflectiveRecord] = []
        trace_instances = list(
            zip(trajectories, eval_batch.scores, eval_batch.outputs, strict=False)
        )

        for trace_instance in trace_instances:
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            kernel_code = traj["kernel_code"]
            compiler_output = traj["compiler_output"]

            if score > 0.0:
                feedback = (
                    "The generated response is correct. The response includes "
                    "the following correct kernel code which compiles:\n\n"
                    f"{kernel_code}"
                )
            else:
                feedback = f"The generated response is incorrect. The following given kernel does not compile:\n\n{kernel_code}\n\nHere is the compiler error that caused this kernel to fail:\n\n{compiler_output}\nThink about what takeaways you can learn from this error to improve your future answers and approach to writing CUDA kernels."

            d: DefaultReflectiveRecord = {
                "Inputs": data["pytorch_kernel"],
                "Generated Outputs": kernel_code,
                "Feedback": feedback,
            }

            print('-------- feedback from compiler: ', d['Feedback'])

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d


adapter = CudaAdapter('openai/kevin-32b')
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset[:5],
    adapter=adapter,
    reflection_lm='openai/gpt-oss-120b',
    max_metric_calls=1,
)

breakpoint()
print(gepa_result)
