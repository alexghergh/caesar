from pydra import Config, REQUIRED

from strategy import Strategy


class CaesarRunConfig(Config):
    def __init__(self):
        # run
        self.run_group = REQUIRED
        self.run_name = REQUIRED

        # dataset
        self.dataset_name = "KernelBench/level1"
        self.num_samples = 1 # samples to generate per problem
                             # essentially parallel scaling with no connections
                             # between directions for now
        self.max_k = 10 # multi-turn

        # LLM configs
        self.model_name = REQUIRED
        self.server_type = REQUIRED
        self.server_address = None
        self.server_port = None

        # decoding parameters
        self.greedy_sample = False
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 50
        self.max_tokens = 4096

        # strategy for prompting; see strategy.py
        self.prompt_strategy = REQUIRED # set on CLI with e.g. prompt_strategy='["BEST_ONLY", "COMPILER_FEEDBACK"]'

        # cpu workers and gpus available
        # workers are number of state machines running at one time
        # set workers to 4x the number of GPU workers or slightly higher
        self.num_workers = 1
        self.num_gpus = 1

        self.gpu_arch = ["Hopper"]  # build for H100 architecture

        # performance evaluation
        self.measure_performance = True
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 600 # time out per round, set to 10 min

        # logging
        self.log_dir_prefix = "/home/8/uc05358/kernel-eval/caesar_log_dir/"
        self.build_dir_prefix = "/home/8/uc05358/kernel-eval/caesar_build_dir/"

        # output verbosity
        self.verbose = False
        self.show_state = False

    def finalize(self):
        # parse strategies from the command line
        if not isinstance(self.prompt_strategy, list):
            raise ValueError("The 'prompt_strategy' variable should be a list of strategies")

        strats = set()
        for elem in self.prompt_strategy:
            strats.add(Strategy[elem])

        self.prompt_strategy = strats

    # server examples

    # deepseek
    #    self.model_name = "deepseek-chat"
    #    self.server_type = "deepseek"
    #    self.temperature = 1.6
    #    self.top_p = 1  # default per API docs

    # anthropic
    #    self.model_name = "claude-3-5-sonnet-20241022" # check this
    #    self.server_type = "anthropic"
    #    self.temperature = 0.8
    #    self.top_p = 1  # default per API docs

    # together.ai
    #    self.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # check this
    #    self.server_type = "together"
    #    self.temperature = 0.7
    #    self.max_tokens = 8192

    # local
    #    self.server_type = "sglang"
    #    self.server_address = "localhost"
    #    self.server_port = 34561

    def __repr__(self):
        return f"CaesarConfig({self.to_dict()})"
