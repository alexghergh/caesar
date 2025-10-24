import os
import queue
import multiprocessing as mp

import torch
import pydra

from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_DATASET, KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_DATASET, KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_DATASET, KERNELBENCH_LEVEL_3_SUBSET_DATASET,
)


from state_machine import CaesarStateMachine
from work import WorkArgs
from custom_transitions import InferenceOnlyNoGPUTransition, InferenceAndGPUTransition
from logger import CaesarLogger
from caesar_config import CaesarRunConfig
from orchestrator import GPUOrchestrator


class MyC(CaesarRunConfig):
    def __init__(self):
        super().__init__()
        self.model_name="lulle"
        self.run_name="lol_name"
        self.run_group="lol_group"
        self.server_type="local"
        # state_machine_strategy="lol"
        # context_strategy="reflection"
        self.use_last_only=True
        self.max_feedback_length=4096
        self.log_dir_prefix="lol_log"
        self.build_dir_prefix="lol_build"
        self.show_state=True
        self.timeout=100000
        self.num_samples=2
        self.num_workers=1 # mp.cpu_count()
        self.max_k=3
        self.verbose=True


dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET,
    "KernelBench/level1-subset": KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    "KernelBench/level2-subset": KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    "KernelBench/level3-subset": KERNELBENCH_LEVEL_3_SUBSET_DATASET,
}


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

    transition = InferenceAndGPUTransition()

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

    # TODO error checking? if something goes bad, maybe catch an exception, and
    # then simply put this back into the work queue (possibly with a fixed
    # number of retries)
    stm = CaesarStateMachine(
        transition,
        config,
        workargs,
        logger,
        process_id=os.getpid(),
        orchestrator=orchestrator,
    )
    stm.run()


def create_work_queue(
    dataset: KernelBenchDataset,
    config: CaesarRunConfig,
) -> mp.Queue:
    """
    Create a work queue of processes.
    """

    # create work list; these are the problems to be solved
    work_list = []
    for problem_id in [23]:# dataset.get_problem_ids():
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


def launch_worker_process(
    config: CaesarRunConfig,
    orchestrator: GPUOrchestrator,
    proc_queue: mp.Queue
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

            # creat and launch process
            work_proc = mp.Process(
                    target=init_and_run_single_sample_work,
                    args=(config, orchestrator, work)
            )
            work_proc.start()

            # TODO timeout this join
            work_proc.join()

            if config.verbose:
                print(f"CPU worker {os.getpid()} finished work {work}")

        except queue.Empty:
            # finished work, queue is empty
            break


@pydra.main(base=MyC)
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
    # - write intermediate logs after each state machine state
    # - better handling of GPU stuff, seems to be broken?

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

    # TODO proper worker queue and stuff, need ID's to track .name is not enough
    # launch CPU workers
    workers_list = []
    for worker in range(config.num_workers):
        worker_proc = mp.Process(
            target=launch_worker_process, args=(config, orchestrator, work_queue),
        )

        if config.verbose:
            print(f"Starting CPU worker with PID {worker_proc.name}")

        worker_proc.start()
        workers_list.append(worker_proc)

    # TODO some form of tracking here (atomic counter)

    # wait for all CPU workers to finish
    for worker_proc in workers_list:
        worker_proc.join()
        if config.verbose:
            print(f"CPU worker {worker_proc.name} finished")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
