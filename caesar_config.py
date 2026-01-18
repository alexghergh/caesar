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
        self.max_turn = 10 # multi-turn

        # LLM configs
        self.model_name = REQUIRED
        self.server_type = REQUIRED
        self.server_address = None
        self.server_port = None

        # decoding parameters
        self.greedy_sample = False
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 50 # doesn't work with all servers
        self.max_tokens = 4096

        # reasoning models setup
        self.reasoning_model = False
        self.reasoning_effort = '' # gpt-5 or gpt-oss only; can be 'low', 'high', 'medium'
        self.reasoning_budget_tokens = 0 # claude models only; if 0, set to 4096

        # prompt feedback options
        self.use_prompt_optimization = False # uses the LLM to create prompts
                                             # from the given information
        self.use_compiler_feedback = True
        self.use_correctness_feedback = True
        self.use_profiler_feedback = True

        # cpu workers and gpus available
        # workers are number of compilation processes running at one time
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

        # rag
        self.rag_docs_dir = "./rag_docs"
        self.rag_index_dir = "./rag_index"
        self.rag_manifest_path = "./rag_index/manifest.json"
        self.rag_top_k = 4
        self.rag_scope = "global"  # or "problem"

        # output verbosity
        self.verbose = False
        self.show_state = False

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

    # local
    #    self.server_type = "sglang"
    #    self.server_address = "localhost"
    #    self.server_port = 34561

    def __repr__(self):
        return f"CaesarConfig({self.to_dict()})"

