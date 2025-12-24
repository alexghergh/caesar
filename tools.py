from langchain.tools import ToolRuntime, tool


# agent context that doesn't change during execution
@dataclass
class CaesarRuntimeContext:
    process_id: int
    config: CaesarRunConfig
    work: WorkArgs
    logger: CaesarLogger
    build_dir: str | os.PathLike
    orchestrator: GPUOrchestrator
    worker_semaphore: mp.Semaphore
    coding_agent: CompiledStateGraph


@tool#(TODO name + description)
def compile(
    kernel_code: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> dict[str, str]:
    config = runtime.context.config
    semaphore = runtime.context.worker_semaphore

    with semaphore:
        # compile kernel and build cache
        returncode, stdout, err = compile_single_sample(
            kernel_src=kernel_code,
            gpu_arch=config.gpu_arch,
            build_dir=runtime.context.build_dir,
            timeout_seconds=config.timeout
        )

    return {
        "return code": returncode,
        "std out": stdout,
        "std err": err,
    }


@tool# (TODO name + description)
def run_kernel(
    reference_problem_python_code: str,
    kernel_code: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> str:
    pass
    config = runtime.context.config
    orchestrator = runtime.context.orchestrator
    work = runtime.context.work

    with orchestrator.reserve_gpu() as gpu_id:

        # launch a separate process to do the GPU work, as each process
        # creates a pytorch context on the GPU; we want to avoid each
        # CPU worker having such a separate context that persists, so
        # spawning a separate process will clear the cache when the
        # process finishes
        result_queue = mp.Queue()
        proc = mp.Process(
            target=evaluate_single_sample_src_mp,
            args=(
                reference_problem_python_code,
                kernel_code,
                config,
                runtime.context.build_dir,
                gpu_id,
                config.timeout,
                result_queue,
            ),
        )
    return result_queue.get()


@tool#(TODO name + description)
def profile_kernel(
    reference_problem_python_code: str,
    kernel_code: str,
    runtime: ToolRuntime[CaesarRuntimeContext]
) -> str:
    config = runtime.context.config
    orchestrator = runtime.context.orchestrator
    work = runtime.context.work

    with orchestrator.reserve_gpu() as gpu_id:
        # launch a separate process to do the GPU work, as each process
        # creates a pytorch context on the GPU; we want to avoid each
        # CPU worker having such a separate context that persists, so
        # spawning a separate process will clear the cache when the
        # process finishes
        result_queue = mp.Queue()
        proc = mp.Process(
            target=get_torch_profiler_info_mp,
            args=(
                reference_problem_python_code,
                kernel_code,
                runtime.context.build_dir,
                gpu_id,
                result_queue,
            ),
        )
        start_time = time.time()
        proc.start()
        proc.join() # wait forever for profiler
        work_time = time.time() - start_time
        result = result_queue.get()

    return result
