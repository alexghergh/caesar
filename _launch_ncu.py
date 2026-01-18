import sys

import torch
from KernelBenchInternal import eval as kernel_eval


def main():
    ref_arch_src, kernel_src, build_dir, gpu_id, seed, num_trials = sys.argv[1:]
    ref_arch_src = ref_arch_src.replace(r'\n', '\n')
    kernel_src = kernel_src.replace(r'\n', '\n')
    gpu_id = int(gpu_id)
    seed = int(seed)
    num_trials = int(num_trials)

    device = torch.device(f"cuda:{gpu_id}")
    kernel_eval.set_seed(seed)

    # load inputs
    context = {}
    _, get_init_inputs, get_inputs = kernel_eval.load_original_model_and_inputs(
        ref_arch_src, context
    )
    init_inputs = get_init_inputs()
    inputs = get_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in init_inputs
    ]
    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    # load and move model
    ModelNew = kernel_eval.load_custom_model(kernel_src, context, build_dir)
    model = ModelNew(*init_inputs).cuda(device=device)
    torch.cuda.synchronize(device=device)

    # run kernel
    for i in range(num_trials):
        _ = model(*inputs)
        torch.cuda.synchronize(device=device)


if __name__ == "__main__":
    main()
