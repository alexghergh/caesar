from dataclasses import dataclass
from typing import Dict, Optional
import pydra
from pydra import Config, REQUIRED
import multiprocessing as mp
import queue
import traceback
import os
import time

from tqdm import tqdm

from logger import CaesarLogger
from states import CaesarState, StateOutcome, Transition, WorkArgs
from state_machine import CaesarStateMachine
from orchestrator import GPUOrchestrator

from utils import check_result_exists, timeout

from transitions_def import MyTransition, Reflection, Reflection_NVCC

from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_DATASET, KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_DATASET, KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_DATASET, KERNELBENCH_LEVEL_3_SUBSET_DATASET,
)

dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET,
}

dataset_name_to_subset_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_SUBSET_DATASET,
}

class CaesarRunConfig(Config):
    def __init__(self):
        # run
        self.run_group = REQUIRED
        self.run_name = REQUIRED
        # dataset
        self.dataset_name = "KernelBench/level1"
        self.level = 1

        self.use_subset = False
        self.num_samples = 1

        # multi-turn stuff
        self.max_k = 10

        # "reflection", "eval_result", "profiler"
        self.context_strategy = ["reflection"]
        # set to False for all previous generations
        # set to True for using only the latest generation
        self.use_last_only = False
        self.state_machine_strategy = "rerun" # default
        self.max_feedback_length = 100000 # in terms of characters, 10k, so much less in tokens

        assert self.state_machine_strategy in ["", "default", "rerun"]

        self.num_workers = 1  # let's do one
        # self.num_gpu_workers for later, assume we get one gpu for each yet

        # LLM configs
        self.model_name = REQUIRED
        self.server_type = REQUIRED
        self.temperature = REQUIRED
        # decoding parameter
        self.greedy_sample = False
        self.temperature = 0.0
        self.top_p = 1.0  # set to consider all tokens
        self.top_k = 50  # set large for default
        self.num_completions = 1
        self.max_tokens = 4096
        # inference server port, only for querying local server
        self.server_address = None
        self.server_port = None

        # Eval Speciifc
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 600 # time out per round, set to 10 min
        self.verbose = False
        self.show_state = False
        self.measure_performance = True


        # Parallel eval settings
        # maximum number of parallel workers at a time
        self.num_workers = 8
        # number of available GPUs
        self.num_gpus = 2


        # if we want to dedicate a GPU to this process
        self.dedicated_gpu_id = 0


        # Logging
        self.log_dir_prefix = "/matx/u/simonguo/kernel_multi_turn/"
        self.build_dir_prefix = "/matx/u/simonguo/kernel_eval_build/"
        self.gpu_arch = ["Ada"]  # build for L40s Ada Lovelace architecture

        self.mock = True
        # testing only 1 sample
        # self.testing = False
        self.debug = False

    def deepseek(self):
        self.model_name = "deepseek-chat"
        self.server_type = "deepseek"
        self.temperature = 1.6
        self.top_p = 1  # default per API docs

    def anthropic(self):
        self.model_name = "claude-3-5-sonnet-20241022" # check this
        self.server_type = "anthropic"
        self.temperature = 0.8
        self.top_p = 1  # default per API docs


    def together(self): # run llama
        self.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # check this
        self.server_type = "together"
        self.temperature = 0.7
        self.max_tokens = 8192

    def test(self):
        # Test logic on one problem one sample one worker one gpu, save to curr dir logs
        self.log_dir_prefix = "./logs"
        self.num_samples = 1
        # self.mock = True
        self.num_workers = 2
        self.num_gpus = 1
        self.debug=True
        self.verbose=False
        self.timeout = 30000 # don't timeout

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"



def start_single_caesar(
    work: WorkArgs,
    config: CaesarRunConfig,
    logger: CaesarLogger,
    process_id: Optional[int] = None,
    orchestrator: Optional[GPUOrchestrator] = None,
) -> int:
    """
    Start a single caesar process
    """

    print(f"Starting Caesar with process_id {process_id} and orchestrator {orchestrator}")
    # define state machine transition configs
    transitions = MyTransition()
    transitions._validate_transitions()

    caesar_01 = CaesarStateMachine(
        transition_cfg=transitions,
        config=config,
        work=work,
        process_id=process_id,
        logger=logger,
        orchestrator=orchestrator,
    )
    returncode = caesar_01.run()
    return returncode

@pydra.main(base=CaesarRunConfig)
def main(
    config: CaesarRunConfig,
    process_id: Optional[int] = None,
    orchestrator: Optional[GPUOrchestrator] = None,
):
    """
    Run a single caesar process, specify a probelm
    """
    print("Running with config", config)

    dataset = KernelBenchDataset(dataset_name=config.dataset_name, level=config.level, use_subset=config.use_subset, dataset=dataset_name_to_dataset[config.dataset_name], subset_dataset=dataset_name_to_subset_dataset[config.dataset_name])
    problem_ids = dataset.get_problem_ids()

    problem_id = 23
    problem = dataset.get_problem_by_id(problem_id)
    sample_id = 0

    orchestrator = GPUOrchestrator(num_gpus=1)
    print(f"Running problem {problem_id} {problem} with sample {sample_id}")
    work = WorkArgs(problem=problem, problem_id=str(problem_id), sample_id=sample_id)

    log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))
    logger = CaesarLogger(log_dir, config, work, verbose=config.verbose, log_name=f"log.json")

    returncode = start_single_caesar(work=work, config=config, logger=logger, process_id=0, orchestrator=orchestrator)

def create_work_queue(config: CaesarRunConfig):
    """
    RN just run everything, if want to add logic to figure out what to run

    problem_id is logically index (1-index)
    should correspond to the problem_id in the dataset
    """

    works :list[WorkArgs] = []
    # Decide which problems we should run

    dataset = KernelBenchDataset(dataset_name=config.dataset_name,
                                 level=config.level,
                                 use_subset=config.use_subset,
                                 dataset=dataset_name_to_dataset[config.dataset_name],
                                 subset_dataset=dataset_name_to_subset_dataset[config.dataset_name])

    if config.debug:
        problem_ids = [23] # just use one problem for debugging
    else:
        problem_ids = dataset.get_problem_ids()

    # # create a list of Work
    for problem_id in problem_ids: # range(10):
        for sample_id in range(config.num_samples):
            if check_result_exists(config.log_dir_prefix, config.run_group, config.run_name, problem_id, sample_id):
                print(f"[SKIP] Result for run {config.run_name} problem {problem_id} sample {sample_id} already exists... skipping")
                continue
            else:
                problem = dataset.get_problem_by_id(problem_id)
                curr_work = WorkArgs(problem=problem, problem_id=str(problem_id), sample_id=sample_id)
                works.append(curr_work)

    print(f"Created {len(works)} works")
    return works

def run_caesar_with_timeout(work, config, process_id, orchestrator, logger, result_queue):
    """Helper function to run caesar in a separate process"""
    try:
        returncode = start_single_caesar(work, config, logger=logger, process_id=process_id, orchestrator=orchestrator)
        result_queue.put(returncode)
    except Exception as e:
        print(f"[Error] Inner process encountered error: {e}")
        traceback.print_exc()
        result_queue.put(1)  # Error code

def worker_process(
        mp_queue: mp.Queue,
        config: CaesarRunConfig,
        progress_queue: mp.Queue,
        process_id: Optional[int] = None,
        orchestrator: Optional[GPUOrchestrator] = None,
    ):
    """
    This is process to be multi-processed
    This gets called in the mp loop as a process. It runs a job
    then grabs a new one from the queue of jobs (atomically).
    """
    while True:
        print(f"[Heartbeat] Process {process_id} start of loop")
        try:
            work = mp_queue.get(timeout=1)
            # if queue is empty it shuts down
            if work is None:
                print(f"[Shutdown] Worker {process_id} received shutdown signal")
                break

            try:
                print(f"[Launch] Worker {process_id} launching work {work}")

                # Initialize logger here.
                log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))
                logger = CaesarLogger(log_dir, config, work, verbose=config.verbose, log_name=f"log.json")

                # Create a queue for the inner process result
                result_queue = mp.Queue()

                # Create and start the inner process
                inner_process = mp.Process(
                    target=run_caesar_with_timeout,
                    args=(work, config, process_id, orchestrator, logger, result_queue)
                )
                inner_process.start()

                try:
                    with timeout(config.timeout * config.max_k):  # 10 minutes per round before timeout
                        inner_process.join()  # Wait for process to complete
                        if inner_process.is_alive():
                            inner_process.terminate()
                            inner_process.join(timeout=1)  # Wait briefly for termination
                            if inner_process.is_alive():
                                inner_process.kill()  # Force kill if still alive
                            raise TimeoutError("Process timed out")

                        # Get the result if process completed
                        returncode = result_queue.get(timeout=1)

                        if returncode == 0:
                            progress_queue.put(1)
                            print(f"[Finish] Worker {process_id} finished work {work}")
                        else:
                            print(f"[Orchestrator] Worker {process_id} encountered error: {returncode}, adding Work {work} back to queue")
                            # Make sure inner process is terminated
                            if inner_process.is_alive():
                                inner_process.terminate()
                                inner_process.join(timeout=1)
                                if inner_process.is_alive():
                                    inner_process.kill()
                            mp_queue.put(work)
                            print(f"Work {work} added back to mp_queue.")

                            continue  # Continue to next work item instead of exiting


                except TimeoutError as e:
                    inner_process.terminate()

                    # TODO: Write to the log.
                    logger._save_timeout_eval("Time limit reached: Kernel either hung or timed out.")

                    print(f"[Orchestrator] Worker {process_id} timed out, shutting down... but adding Work {work} back to queue")
                    mp_queue.put(work)

                    raise TimeoutError(f"Process timed out: {e}")

            except TimeoutError as e:
                print(f"[Timeout] Worker {process_id} GPU acquisition timed out: {e}")

                # Write a timeout
                log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))

                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "TIMEOUT"), "w") as f:
                    pass
                # Optionally requeue the work item
                # mp_queue.put(work)
                continue

            except Exception as e:
                print(f"[Error] Worker {process_id} encountered error: {e}")
                traceback.print_exc()
                # Optionally log or handle specific errors differently
                continue
        except queue.Empty:
            # if queue is empty it shuts down
            time.sleep(10)
            continue
            # break


@pydra.main(base=CaesarRunConfig)
def main_orchestrator(config: CaesarRunConfig):
    print(f"[Orchestrator] Starting orchestrator for Run {config.run_name}")
    print(f"[Orchestrator] Num workers: {config.num_workers} Num GPUs: {config.num_gpus}")
    mp.set_start_method('spawn', force=True)

    orchestrator = GPUOrchestrator(num_gpus=config.num_gpus)

    # Create multiple worker processes
    processes = []
    job_queue = mp.Queue()
    progress_queue = mp.Queue()  # New queue for progress tracking
    total_work = create_work_queue(config)
    total_work_count = len(total_work)

    if total_work_count == 0:
        print("[Orchestrator] No work to do, shutting down and exiting...")
        return

    # populate the queue with work
    for work in total_work:
        job_queue.put(work)

    try:
        # Start worker processes
        for worker_id in range(config.num_workers):
            p = mp.Process(
                target=worker_process,
                args=(job_queue, config, progress_queue),
                kwargs={"process_id": worker_id, "orchestrator": orchestrator}
            )
            p.start()
            processes.append(p)

        # Progress bar for overall progress
        with tqdm(total=total_work_count, desc="Overall Progress") as pbar:
            completed_work = 0
            while completed_work < total_work_count:
                progress_queue.get()  # Wait for a work item to complete
                completed_work += 1
                pbar.update(1)

        # Wait for all processes to complete
        for p in processes:
            p.terminate()
            p.join()

    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down...")
        # Terminate all processes
        for p in processes:
            p.terminate()
            p.join()


if __name__ == "__main__":
    main_orchestrator()
    # main()

