import multiprocessing as mp
import os
import time

from caesar_config import CaesarRunConfig
from orchestrator import GPUOrchestrator
from work import WorkArgs

try:
    import pydra
except ModuleNotFoundError:
    pydra = None


KERNEL_BENCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "KernelBench")
)


def _load_dataset_registry():
    from KernelBenchInternal.dataset import (
        CUDAFORGE_SUBSET,
        KERNELBENCH_LEVEL_1_DATASET,
        KERNELBENCH_LEVEL_1_RANDOM_SUBSET_DATASET,
        KERNELBENCH_LEVEL_1_SUBSET_DATASET,
        KERNELBENCH_LEVEL_2_DATASET,
        KERNELBENCH_LEVEL_2_RANDOM_SUBSET_DATASET,
        KERNELBENCH_LEVEL_2_SUBSET_DATASET,
        KERNELBENCH_LEVEL_3_DATASET,
        KERNELBENCH_LEVEL_3_RANDOM_SUBSET_DATASET,
        KERNELBENCH_LEVEL_3_SUBSET_DATASET,
        KERNELBENCH_LEVELS_12_REPRESENTATIVE_DATASET,
        METASTARK_SUBSET,
        KernelBenchDataset,
    )

    dataset_name_to_dataset = {
        "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
        "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
        "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET,
        "KernelBench/level1-subset": KERNELBENCH_LEVEL_1_SUBSET_DATASET,
        "KernelBench/level2-subset": KERNELBENCH_LEVEL_2_SUBSET_DATASET,
        "KernelBench/level3-subset": KERNELBENCH_LEVEL_3_SUBSET_DATASET,
        "KernelBench/level1-random": KERNELBENCH_LEVEL_1_RANDOM_SUBSET_DATASET,
        "KernelBench/level2-random": KERNELBENCH_LEVEL_2_RANDOM_SUBSET_DATASET,
        "KernelBench/level3-random": KERNELBENCH_LEVEL_3_RANDOM_SUBSET_DATASET,
        "KernelBench/levels12-subset": KERNELBENCH_LEVELS_12_REPRESENTATIVE_DATASET,
        "KernelBench/cudaforge-subset": CUDAFORGE_SUBSET,
        "KernelBench/metastark-subset": METASTARK_SUBSET,
        # debug
        "KernelBench/level1-test": [
            os.path.join(KERNEL_BENCH_PATH, "KernelBench", "level1", "23_Softmax.py")
        ],
    }

    return KernelBenchDataset, dataset_name_to_dataset


def launch_state_machine_process(
    config: CaesarRunConfig,
    orchestrator: GPUOrchestrator,
    work: WorkArgs,
    progress: mp.Value,
    proc_sem: mp.Semaphore,
) -> None:
    """
    Launch a state machine process. This is meant to be launched in a
    multiprocessing context. Each worker works on a problem, with all its
    samples.
    """
    from state_machine import run_state_machine

    run_state_machine(os.getpid(), config, work, orchestrator, progress, proc_sem)


def _run_main(config: CaesarRunConfig) -> None:
    missing_dependencies = []
    for module_name in ("torch", "tqdm", "KernelBenchInternal"):
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            missing_dependencies.append(module_name)

    if missing_dependencies:
        missing = ", ".join(sorted(missing_dependencies))
        print(
            f"Missing required runtime dependencies: {missing}. "
            "Install project dependencies and re-run."
        )
        return

    from tqdm import tqdm

    KernelBenchDataset, dataset_name_to_dataset = _load_dataset_registry()

    if config.verbose:
        print("Running with config: ", config)

    dataset = KernelBenchDataset(
        dataset_name_to_dataset.get(
            config.dataset_name,
            dataset_name_to_dataset["KernelBench/level1"],
        )
    )

    if config.verbose:
        print(f"There are {len(dataset) * config.num_samples} total samples to solve")

    orchestrator = GPUOrchestrator(num_gpus=config.num_gpus, verbose=config.verbose)

    progress = mp.Value("i", 0, lock=True)
    proc_sem = mp.Semaphore(value=config.num_workers)

    workers_list = []
    for problem_id in dataset.get_problem_ids():
        workargs = WorkArgs(
            problem_id=problem_id,
            sample_id=-1,
            problem_path="",
        )
        workargs.problem_path = dataset.get_problem_path_by_id(workargs.problem_id)

        worker_proc = mp.Process(
            target=launch_state_machine_process,
            args=(config, orchestrator, workargs, progress, proc_sem),
        )

        worker_proc.start()
        workers_list.append(worker_proc)

    with tqdm(
        total=len(dataset) * config.num_samples,
        desc="Overall progress (per sample)",
        miniters=1,
    ) as pbar:
        while pbar.n != len(dataset) * config.num_samples:
            time.sleep(1)
            with progress.get_lock():
                pbar.update(progress.value)
                progress.value = 0

    for worker_proc in workers_list:
        worker_proc.join()


if pydra is not None:
    @pydra.main(base=CaesarRunConfig)
    def main(config: CaesarRunConfig):
        _run_main(config)
else:
    def main(config: CaesarRunConfig | None = None):
        if config is None:
            print(
                "pydra is not installed. Install project dependencies to run this CLI."
            )
            return
        _run_main(config)


if __name__ == "__main__":
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None:
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    main()
