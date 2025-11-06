import os
import time
import queue
import multiprocessing as mp

import torch
import pydra
from tqdm import tqdm

from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_DATASET, KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_DATASET, KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_DATASET, KERNELBENCH_LEVEL_3_SUBSET_DATASET,
)


from state_machine import CaesarStateMachine
from work import WorkArgs
from custom_transitions import InferenceAndGPUAndProfilerTransition
from logger import CaesarLogger
from caesar_config import CaesarRunConfig
from orchestrator import GPUOrchestrator


dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET,
    "KernelBench/level1-subset": KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    "KernelBench/level2-subset": KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    "KernelBench/level3-subset": KERNELBENCH_LEVEL_3_SUBSET_DATASET,

    # debug
    "KernelBench/level1-test": ["../KernelBench/KernelBench/level1/23_Softmax.py"],
}


def create_work_queue(
    dataset: KernelBenchDataset,
    config: CaesarRunConfig,
) -> mp.Queue:
    """
    Create a work queue of processes.
    """

    # create work list; these are the problems to be solved
    work_list = []
    for problem_id in dataset.get_problem_ids():
        for sample_id in range(1, config.num_samples + 1):
            workargs = WorkArgs(problem_id=problem_id, sample_id=sample_id, problem_path="")
            workargs.problem_path = dataset.get_problem_path_by_id(workargs.problem_id)

            work_list.append(workargs)

    if config.verbose:
        print(f"Created {len(work_list)} problems to solve")

    # add these to a global work queue across workers
    work_queue = mp.Queue()
    for work in work_list:
        work_queue.put(work)

    return work_queue


def init_and_run_single_sample_work(
    config: CaesarRunConfig,
    orchestrator: GPUOrchestrator,
    workargs: WorkArgs,
) -> None:
    """
    Create the work object for a single problem, for a single sample, and
    run the state machine.
    Initializes the work args, a logger and a state machine.
    """

    transition = InferenceAndGPUAndProfilerTransition()

    logger = CaesarLogger(
        os.path.join(
            config.log_dir_prefix,
            config.run_group,
            config.run_name,
            workargs.get_log_path(),
        ),
        config,
        workargs,
    )

    stm = CaesarStateMachine(
        transition,
        config,
        workargs,
        logger,
        process_id=os.getpid(),
        orchestrator=orchestrator,
    )
    stm.run()


def launch_worker_process(
    config: CaesarRunConfig,
    orchestrator: GPUOrchestrator,
    proc_queue: mp.Queue,
    progress: mp.Value,
) -> None:
    """
    Launch a worker process. This is meant to be launched in a multiprocessing
    context. Each worker will get work from the passed queue, and will execute
    it. If the queue is empty, return.
    """

    while True:
        try:
            work = proc_queue.get(block=True, timeout=1) # timeout means empty queue

            if config.verbose:
                print(f"CPU worker {os.getpid()} starting work {work}")

            # create and launch process
            work_proc = mp.Process(
                    target=init_and_run_single_sample_work,
                    args=(config, orchestrator, work)
            )
            work_proc.start()
            work_proc.join()

            # update global progress
            with progress.get_lock():
                progress.value += 1

            if config.verbose:
                print(f"CPU worker {os.getpid()} finished work {work}")

        except queue.Empty:
            # finished work, queue is empty
            break


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

    if config.verbose:
        print("Running with config: ", config)

    dataset = KernelBenchDataset(
        dataset_name_to_dataset.get(config.dataset_name, "KernelBench/level1")
    )

    # global, for all problems
    orchestrator = GPUOrchestrator(
        num_gpus=torch.cuda.device_count(), verbose=config.verbose
    )

    # global work queue
    work_queue = create_work_queue(dataset, config)

    # track global problem progress
    progress = mp.Value('i', 0, lock=True)

    # launch CPU workers
    workers_list = []
    for worker in range(config.num_workers):
        worker_proc = mp.Process(
            target=launch_worker_process,
            args=(config, orchestrator, work_queue, progress),
        )

        if config.verbose:
            print(f"Starting CPU worker with PID {worker_proc.name}")

        worker_proc.start()
        workers_list.append(worker_proc)

    # tqdm progress tracker
    with tqdm(total=work_queue.qsize(), desc="Overall progress") as pbar:
        while not work_queue.empty():
            time.sleep(1)
            pbar.n = progress.value
            pbar.last_print_n = progress.value
            pbar.update(0)

    # wait for all CPU workers to finish
    for worker_proc in workers_list:
        worker_proc.join()
        if config.verbose:
            print(f"CPU worker {worker_proc.name} finished")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
