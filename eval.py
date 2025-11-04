import os
import torch
import multiprocessing as mp

from torch.profiler import profile, ProfilerActivity

from KernelBenchInternal import eval as kernel_eval
from KernelBenchInternal import utils as kernel_utils

from caesar_config import CaesarRunConfig
from utils import Timeout


def get_kernel_hash(kernel_src: str) -> str:
    return str(hash(kernel_src))


def compile_single_sample(kernel_src: str,
                          config: CaesarRunConfig,
                          build_dir: str,
                          timeout_seconds: int = 480) -> tuple[int, str, str]:
    """
    Compile kernel on CPU and capture errors.
    """
    kernel_utils.set_gpu_arch(config.gpu_arch)

    # the build dir contains .c, .cu, .so, .o, .py and some ninja files
    # some of these need to be kept as cache
    kernel_hash = get_kernel_hash(kernel_src)
    kernel_build_dir = os.path.join(build_dir, kernel_hash)

    try:
        with Timeout(timeout_seconds):
            returncode, stdout, err = kernel_eval.build_compile_cache_with_capturing(
                custom_model_src=kernel_src,
                verbose=False,
                build_dir=kernel_build_dir,
            )
            return returncode, stdout, err
    except TimeoutError:
        print(f"[WARNING] Compilation timed out after {timeout_seconds} seconds")
        return -1, f"Compilation timed out after {timeout_seconds} seconds", f"Compilation timed out after {timeout_seconds} seconds"
    except Exception as e:
        print(f"[WARNING] Last level catch when CPU pre-compiling kernel: Some issue while compiling and attempting to cache for kernel: {e} ")
        return -1, str(e), str(e)


def evaluate_single_sample_src(ref_arch_src: str,
                               kernel_src: str,
                               configs: CaesarRunConfig,
                               build_dir: str,
                               gpu_id: int,
                               timeout_seconds: int = 480,
                               ) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code against a reference architecture source
    code.
    """
    kernel_hash = get_kernel_hash(kernel_src)
    build_dir = os.path.join(build_dir, kernel_hash)

    import torch
    device = torch.device(f"cuda:{gpu_id}")

    try:
        with Timeout(timeout_seconds):
            eval_result = kernel_eval.eval_kernel_against_ref(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                measure_performance=True,
                verbose=False,
                num_correct_trials=configs.num_correct_trials,
                num_perf_trials=configs.num_perf_trials,
                build_dir=build_dir,
                device=device
            )
            return eval_result
    except TimeoutError:
        print(f"[WARNING] Evaluation timed out after {timeout_seconds} seconds")
        metadata = {
            "timeout_error": f"Evaluation timed out after {timeout_seconds} seconds",
            "hardware": torch.cuda.get_device_name(device=device),
            "device": str(device)
        }
        metadata = kernel_eval.check_metadata_serializable(metadata)
        eval_result = kernel_eval.KernelExecResult(
            compiled=False, correctness=False, metadata=metadata
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e):
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {
                "cuda_error": f"CUDA Error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device)
            }

            metadata = kernel_eval.check_metadata_serializable(metadata)
            eval_result = kernel_eval.KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result
        else:
            metadata = {
                "other_error": f"error: {str(e)}",
                "hardware": torch.cuda.get_device_name(device=device),
                "device": str(device)
            }
            metadata = kernel_eval.check_metadata_serializable(metadata)
            eval_result = kernel_eval.KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            return eval_result


def evaluate_single_sample_src_mp(ref_arch_src: str,
                                  kernel_src: str,
                                  configs: CaesarRunConfig,
                                  build_dir: str,
                                  gpu_id: int,
                                  timeout_seconds: int,
                                  result_queue: mp.Queue) -> None:
    """
    Same as `evaluate_single_sample_src`, but meant to be called in a
    multiprocessing context. Instead of returning the result, it puts it in a
    queue passed as parameter.
    """
    result_queue.put(evaluate_single_sample_src(ref_arch_src,
                                                kernel_src,
                                                configs,
                                                build_dir,
                                                gpu_id,
                                                timeout_seconds))


def get_torch_profiler_info(ref_arch_src: str,
                            kernel_src: str,
                            build_dir: str,
                            gpu_id: int,
                            num_trials: int = 100,
                            table_row_limit: int = 10,
                            seed_num: int = 42) -> str:
    """
    Get the profiler info for a particular kernel.

    Notes about profiling:
        - We do not set p.toggle_collection_dynamic explicitly,
        - We only collect CUDA activity (ProfilerActivity.CUDA), as we are only
          interested in the kernel.
    """
    kernel_hash = get_kernel_hash(kernel_src)
    build_dir = os.path.join(build_dir, kernel_hash)

    # get the inputs and problem size
    context = {}
    _, get_init_inputs, get_inputs = kernel_eval.load_original_model_and_inputs(
        ref_arch_src, context
    )

    device = torch.device(f"cuda:{gpu_id}")
    kernel_eval.set_seed(seed_num)

    inputs = get_inputs()
    init_inputs = get_init_inputs()
    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in init_inputs
    ]

    # construct the model to profile
    ModelNew = kernel_eval.load_custom_model(kernel_src, context, build_dir)
    model = ModelNew(*init_inputs)
    torch.cuda.synchronize(device=device)

    model = model.cuda(device=device)

    with torch.no_grad():
        profiling_scheduler = torch.profiler.schedule(
            skip_first=2,
            wait=2,
            warmup=3,
            active=num_trials,
        )

        with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=profiling_scheduler,
        ) as prof:
            for _ in range(num_trials):
                _ = model(*inputs)
                prof.step()

        profiler_output = prof.key_averages().table(sort_by='cuda_time_total',
                                                    row_limit=table_row_limit)
    return profiler_output


def get_torch_profiler_info_mp(ref_arch_src: str,
                               kernel_src: str,
                               build_dir: str,
                               gpu_id: int,
                               result_queue: mp.Queue) -> None:
    """
    Same as `get_torch_profiler_info`, but meant to be called in a
    multiprocessing context. It puts the result in a queue instead of returning.
    """
    result_queue.put(get_torch_profiler_info(ref_arch_src,
                                             kernel_src,
                                             build_dir,
                                             gpu_id))
