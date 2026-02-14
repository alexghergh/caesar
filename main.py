import os
import time
import multiprocessing as mp

import torch
import pydra
from tqdm import tqdm

from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_DATASET,
    KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    KERNELBENCH_LEVEL_1_RANDOM_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_DATASET,
    KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_RANDOM_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_DATASET,
    KERNELBENCH_LEVEL_3_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_RANDOM_SUBSET_DATASET,
    KERNELBENCH_LEVELS_12_REPRESENTATIVE_DATASET,
    CUDAFORGE_SUBSET,
    METASTARK_SUBSET,
)

from state_machine import run_state_machine
from work import WorkArgs
from caesar_config import CaesarRunConfig
from orchestrator import GPUOrchestrator


KERNEL_BENCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "KernelBench")
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
    # launch state machine
    run_state_machine(os.getpid(), config, work, orchestrator, progress, proc_sem)


@pydra.main(base=CaesarRunConfig)
def main(config: CaesarRunConfig):
    # TODOs:
    # - right now, the samples per problem are independent (i.e. each separately
    # queries the model and continues iterating); in the (near) future i want to
    # be able to have best-k selection, i.e. after each round, pool together the
    # best k/total samples, then randomly distribute those best-k and start the
    # next round from there; in theory, it looks like you just need to open the
    # config files for all samples, pick the best, then re-write the config
    # files; anything else to consider? There will be stalls and dependencies if
    # some samples did not finish
    # - CoT/ICL examples of progressive optimization
    # - RAG
    # - hardware architecture information
    # - state machine run orchestrator
    # - ncu / nsys profiling instead of torch

    if config.verbose:
        print("Running with config: ", config)

    dataset = KernelBenchDataset(
        dataset_name_to_dataset.get(config.dataset_name, "KernelBench/level1")
    )

    if config.verbose:
        print(f"There are {len(dataset) * config.num_samples} total samples to solve")

    # global, for all problems
    orchestrator = GPUOrchestrator(
        num_gpus=config.num_gpus, verbose=config.verbose
    )

    # track global problem progress
    progress = mp.Value('i', 0, lock=True)

    # semaphore to limit process launching (this only applies to each state
    # machine launching sub-processes to solve samples)
    proc_sem = mp.Semaphore(value=config.num_workers)

    # launch state machine workers, 1 per problem
    workers_list = []
    for problem_id in dataset.get_problem_ids():

        # create work args
        workargs = WorkArgs(
            problem_id=problem_id,
            sample_id=-1,
            problem_path="",
        )
        workargs.problem_path = dataset.get_problem_path_by_id(workargs.problem_id)

        # launch state machine process
        worker_proc = mp.Process(
            target=launch_state_machine_process,
            args=(config, orchestrator, workargs, progress, proc_sem),
        )

        worker_proc.start()
        workers_list.append(worker_proc)

    # tqdm progress tracker
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

    # wait for all state machine workers to finish
    for worker_proc in workers_list:
        worker_proc.join()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
