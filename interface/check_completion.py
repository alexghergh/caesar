from caesar.utils import get_run_group_stats
from caesar.interface.run_mapping import (
    RUN_MAPPING_LEVEL_1,
    RUN_MAPPING_LEVEL_2,
    RUN_MAPPING_LEVEL_3
)
from tabulate import tabulate

def check_completion_status(log_dir_prefix: str):
    """Check completion status for all runs in the mapping."""
    
    # Process all level mappings
    all_mappings = {
        "Level 1": RUN_MAPPING_LEVEL_1,
        "Level 2": RUN_MAPPING_LEVEL_2,
        "Level 3": RUN_MAPPING_LEVEL_3
    }
    
    for level_name, mapping in all_mappings.items():
        table_data = []
        headers = ["Strategy", "Model", "Run Group", "Run Name", "Completed Evaluations"]
        
        # Collect data for each strategy and model
        for strategy, models in mapping.items():
            for model_name, run_info in models.items():
                run_group = run_info["run_group"]
                run_name = run_info["run_name"]
                stats = get_run_group_stats(log_dir_prefix, run_group)
                
                completed_evals = stats.get(run_name, "No evaluations found")
                table_data.append([
                    strategy,
                    model_name,
                    run_group,
                    run_name,
                    completed_evals
                ])
        
        # Print level header and table
        print(f"\n{level_name} Completion Status:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    # You might want to make this configurable via command line args
    LOG_DIR_PREFIX = "/matx/u/simonguo/kernel_multi_turn/"
    check_completion_status(LOG_DIR_PREFIX)


    