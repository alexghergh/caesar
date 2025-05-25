"""
Mapping of Caesar Runs with Various

Each Level has a mapping object

Each mapping object (dict)
- indexed by strategy
- and then indexed by model
- and in the inner most dict, has the run group and run name info

Every run here are done with temperature 0.0
"""

# You can modify this mapping to include your runs.
RUN_MAPPING_LEVEL_1 = {
    # "reflection_all_prev": {
    #     "deepseek-v3": {
    #         "run_group": "level1_reflection_all_prev_deepseek",
    #         "run_name": "run_deepseek_turns_10"
    #     },
    #     "llama-3.1-70b-inst": {
    #         "run_group": "level1_reflection_all_prev_llama",
    #         "run_name": "run_llama_turns_10"
    #     },
    #     "deepseek-R1": {
    #         "run_group": "level1_reflection_all_prev_deepseek",
    #         "run_name": "run_v0_deepseek_r1_turn"
    #     },
    # },
    "reflection_last_only": {
        "deepseek-v3": {
            "run_group": "level1_reflection_last_only_deepseek",
            "run_name": "run_deepseek_turns_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level1_reflection_last_only_llama", 
            "run_name": "run_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level1_reflection_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn"
        },
    },
    "eval_result_last_only": {
        "deepseek-v3": {
            "run_group": "level1_eval_result_last_only_deepseek",
            "run_name": "run_v0_deepseek_turns_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level1_eval_result_last_only_llama",
            "run_name": "run_v0_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level1_eval_result_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn_10"
        },
    },
    "eval_result_profiler_last_only": {
        "deepseek-v3": {
            "run_group": "level1_eval_result_profiler_last_only_deepseek",
            "run_name": "run_v0_deepseek_turn_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level1_eval_result_profiler_last_only_llama",
            "run_name": "run_v0_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level1_eval_result_profiler_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn" #oops i forgot the name 10
        },
    }
}

RUN_MAPPING_LEVEL_2 = {
    # "reflection_all_prev": {
    #     "deepseek-v3": {
    #         "run_group": "level2_reflection_all_prev_deepseek",
    #         "run_name": "run_deepseek_turns_10"
    #     },
    #     "llama-3.1-70b-inst": {
    #         "run_group": "level2_reflection_all_prev_llama",
    #         "run_name": "run_llama_turns_10"
    #     },
    #     "deepseek-R1": {
    #         "run_group": "level2_reflection_all_prev_deepseek",
    #         "run_name": "run_v0_deepseek_r1_turn"
    #     },
    # },
    "reflection_last_only": {
        "deepseek-v3": {
            "run_group": "level2_reflection_last_only_deepseek",
            "run_name": "run_deepseek_turns_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level2_reflection_last_only_llama", 
            "run_name": "run_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level2_reflection_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn"
        },
    },
    "eval_result_last_only": {
        "deepseek-v3": {
            "run_group": "level2_eval_result_last_only_deepseek",
            "run_name": "run_v0_deepseek_turns_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level2_eval_result_last_only_llama",
            "run_name": "run_v0_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level2_eval_result_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn_10"
        },
    },
    "eval_result_profiler_last_only": {
        "deepseek-v3": {
            "run_group": "level2_eval_result_profiler_last_only_deepseek",
            "run_name": "run_v0_deepseek_turn_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level2_eval_result_profiler_last_only_llama",
            "run_name": "run_v0_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level2_eval_result_profiler_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn"
        },
    }
}

RUN_MAPPING_LEVEL_3 = {
    # "reflection_all_prev": {
    #     "deepseek-v3": {
    #         "run_group": "level3_reflection_all_prev_deepseek",
    #         "run_name": "run_deepseek_turns_10"
    #     },
    #     "llama-3.1-70b-inst": {
    #         "run_group": "level3_reflection_all_prev_llama",
    #         "run_name": "run_llama_turns_10"
    #     },
    #     # THIS TOOK FOREVER TO RUN COMMENT OUT FOR ANALYSIS
    #     "deepseek-R1": {
    #         "run_group": "level3_reflection_all_prev_deepseek",
    #         "run_name": "run_v0_deepseek_r1_turn"
    #     },
    # },
    "reflection_last_only": {
        "deepseek-v3": {
            "run_group": "level3_reflection_last_only_deepseek",
            "run_name": "run_deepseek_turns_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level3_reflection_last_only_llama", 
            "run_name": "run_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level3_reflection_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn"
        },
    },
    "eval_result_last_only": {
        "deepseek-v3": {
            "run_group": "level3_eval_result_last_only_deepseek",
            "run_name": "run_v0_deepseek_turns_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level3_eval_result_last_only_llama",
            "run_name": "run_v0_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level3_eval_result_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn_10"
        },
    },
    "eval_result_profiler_last_only": {
        "deepseek-v3": {
            "run_group": "level3_eval_result_profiler_last_only_deepseek",
            "run_name": "run_v0_deepseek_turn_10"
        },
        "llama-3.1-70b-inst": {
            "run_group": "level3_eval_result_profiler_last_only_llama",
            "run_name": "run_v0_llama_turns_10"
        },
        "deepseek-R1": {
            "run_group": "level3_eval_result_profiler_last_only_deepseek",
            "run_name": "run_v0_deepseek_r1_turn"
        },
    }
}



