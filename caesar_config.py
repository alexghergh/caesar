from pydra import Config, REQUIRED


class CaesarRunConfig(Config):
    def __init__(self):
        # run
        self.run_group : str = REQUIRED
        self.run_name : str = REQUIRED

        # dataset
        self.dataset_name = "KernelBench/level1"
        self.num_samples = 1 # samples to generate per problem
                            # essentially parallel scaling with no connections between directions for now

        # LLM configs
        self.model_name = REQUIRED
        self.server_type = REQUIRED
        self.server_address = None
        self.server_port = None

        # decoding parameters
        self.temperature = REQUIRED
        self.greedy_sample = False
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 50
        self.num_completions = 1 # TODO what is this? beam search?
        self.max_tokens = 4096

        # multi-turn
        self.max_k = 10

        # TODO rework this
        # whether to include generated kernels so far into the current turn
        # set to False for all previous generations
        # set to True for using only the latest generation
        self.use_last_only = True

        # TODO
        # # "reflection", "eval_result", "profiler"
        # self.context_strategy = ["reflection"]
        # self.state_machine_strategy = "rerun" # default
        # self.max_feedback_length = 100000 # in terms of characters, 10k, so much less in tokens

        # assert self.state_machine_strategy in ["", "default", "rerun"]

        # cpu workers and gpus available
        # workers are number of state machines running at one time
        # set workers to the number of gpus or slightly higher
        self.num_workers = 1
        self.num_gpus = 1

        self.gpu_arch = ["Hopper"]  # build for H100 architecture
        # TODO make the above architecture some form of enum

        # performance evaluation
        self.measure_performance = True
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 600 # time out per round, set to 10 min

        # logging
        # should these just be REQUIRED?
        self.log_dir_prefix = "/home/8/uc05358/kernel-eval/caesar_log_dir/"
        self.build_dir_prefix = "/home/8/uc05358/kernel-eval/caesar_build_dir/"

        # output verbosity
        self.verbose = False
        self.show_state = False

    # TODO all the below?
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

    def together(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # check this
        self.server_type = "together"
        self.temperature = 0.7
        self.max_tokens = 8192

    def local(self):
        self.server_type = "sglang"
        self.server_address = "localhost"
        self.server_port = 34561
        # TODO what else? model_name is dynamic

    def __repr__(self):
        return f"CaesarConfig({self.to_dict()})"
