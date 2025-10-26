import multiprocessing as mp
import random
import time
import signal
import sys
from contextlib import contextmanager


class GPUOrchestrator:
    """
    Orchestrator for concurrent GPU usage.

    This is used to ensure that we don't have multiple processes
    accessing the same GPU.
    """
    def __init__(self, num_gpus=8, verbose=False):
        self.num_gpus = num_gpus

        # semaphore to limit total GPU access
        self.gpu_semaphore = mp.Semaphore(num_gpus)

        # track which GPUs are in use
        self.gpu_status = mp.Array("i", [0] * num_gpus, lock=True)

        self.verbose = verbose

        if self.verbose:
            print(f"[ORCHESTRATOR] GPU Orchestrator initialized with {num_gpus} GPUs")

        # create a listener for cleanup on shutdown
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

    def get_available_gpu(self):
        """Find and reserve an available GPU."""
        with self.gpu_status:
            for i in range(self.num_gpus):
                if self.gpu_status[i] == 0:
                    self.gpu_status[i] = 1
                    return i
        return None

    def release_gpu(self, gpu_id):
        """Release a GPU back to the pool."""
        with self.gpu_status:
            self.gpu_status[gpu_id] = 0

    @contextmanager
    def reserve_gpu(self):
        """Context manager for GPU reservation."""
        self.gpu_semaphore.acquire(block=True)
        gpu_id = self.get_available_gpu()
        try:
            yield gpu_id
        finally:
            self.release_gpu(gpu_id)
            self.gpu_semaphore.release()

    def cleanup(self, *args):
        if self.verbose:
            print("\n[ORCHESTRATOR] Cleaning up GPU Orchestrator...")
        sys.exit(0)



def worker_process(process_id, orchestrator):
    """Simulated worker process that needs GPU access."""
    while True:
        # Simulate some CPU work before needing GPU
        time.sleep(random.uniform(0.1, 2))

        print(f"[Orchestration] Process {process_id} requesting GPU...")

        with orchestrator.reserve_gpu() as gpu_id:
            print(f"[Orchestration] Process {process_id} acquired GPU {gpu_id}")

            # Simulate GPU work
            try:
                # device = torch.device(f"cuda:{gpu_id}")
                # Simulate some GPU computation
                work_time = random.uniform(1, 5)
                print(
                    f"[Orchestration] Process {process_id} working on GPU {gpu_id} for {work_time:.2f} seconds"
                )
                time.sleep(work_time)
            except Exception as e:
                print(f"Process {process_id} encountered error: {e}")

            print(f"[Orchestration] Process {process_id} releasing GPU {gpu_id}")


def main():
    # Create the GPU orchestrator
    orchestrator = GPUOrchestrator(num_gpus=8)

    # Create multiple worker processes
    num_workers = 12  # More workers than GPUs to demonstrate queuing
    processes = []

    try:
        # Start worker processes
        for i in range(num_workers):
            p = mp.Process(target=worker_process, args=(i, orchestrator))
            p.start()
            processes.append(p)

        # Wait for all processes to complete (they won't in this case, need manual interrupt)
        for p in processes:
            p.join()

    except KeyboardInterrupt:
        print("\nShutting down...")
        # Terminate all processes
        for p in processes:
            p.terminate()
            p.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
