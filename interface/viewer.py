import os
import re
import sys
import json
import traceback
import subprocess
from pathlib import Path

from fasthtml.common import (
    fast_app,
    serve,
    Titled,
    Script,
    Div,
    Span,
    Form,
    Select,
    Option,
    Input,
    Label,
    Button,
    Details,
    Summary,
    Pre,
    P,
    A,
    H2,
    Code,
    Table,
    Tr,
    Th,
    Td,
)
from KernelBenchInternal.dataset import (
    KernelBenchDataset,
    KERNELBENCH_LEVEL_1_DATASET, KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    KERNELBENCH_LEVEL_2_DATASET, KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    KERNELBENCH_LEVEL_3_DATASET, KERNELBENCH_LEVEL_3_SUBSET_DATASET,
)

# from run_mapping import (
#     RUN_MAPPING_LEVEL_1,
#     RUN_MAPPING_LEVEL_2,
#     RUN_MAPPING_LEVEL_3
# )


# get root caesar directory (i.e., the parent of 'interface')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# add sys paths (needed for python import discovery)
sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'analysis'))

from utils import (
    get_available_run_groups,
    get_available_runs,
    get_available_problem_ids,
    get_run_group_finished_runs,
    get_prev_problem_id,
    get_next_problem_id,
    load_json_data,
    #
    get_turn_trajectory_overviews,
    get_turns,
    get_available_runs,
    fetch_baseline_time_by_problem_id,
)


BASE_LOG_DIR = os.path.join(ROOT_DIR, "caesar_log_dir") # path to logs

dataset_name_to_dataset = {
    "KernelBench/level1": KERNELBENCH_LEVEL_1_DATASET,
    "KernelBench/level2": KERNELBENCH_LEVEL_2_DATASET,
    "KernelBench/level3": KERNELBENCH_LEVEL_3_DATASET,
    "KernelBench/level1-subset": KERNELBENCH_LEVEL_1_SUBSET_DATASET,
    "KernelBench/level2-subset": KERNELBENCH_LEVEL_2_SUBSET_DATASET,
    "KernelBench/level3-subset": KERNELBENCH_LEVEL_3_SUBSET_DATASET,
}

# TODO
# # Add these constants before the route definitions
LEVELS = ["level1", "level2", "level3"]
STRATEGIES = [
    "reflection_all_prev",
    "reflection_last_only",
    "eval_result_last_only",
    "eval_result_profiler_last_only"
]
MODELS = [
    "deepseek-v3",
    "llama-3.1-70b-inst",
    "deepseek-R1"
]

# TODO
# get timing comparisons
# baseline_time_filepath = os.path.join(KERNELBENCH_REPO_DIR, "KernelBench", "results", "timing", "H100_tsubame", "baseline_time_torch.json")
# baseline_torch_compile_time_filepath = os.path.join(PATH_TO_REPO_DIR, "KernelBench", "results", "timing", "H100_tsubame", "baseline_time_torch_compile_inductor_default.json")


# fastapi config
PORT = 5008
HOST_IP = 'localhost'
print(f"[INFO] Starting Viewer at {HOST_IP}:{PORT}")

app, rt = fast_app()

# @route('/update_problem_list')
# def get(run_name: str):
#     """Update problem list when run is selected"""
#     return Input(
#         type="number",
#         name="problem_id",
#         id="problem-select",
#         placeholder="Problem ID",
#         min="1",
#         hx_get="/update_sample_list",
#         hx_target="#sample-container",
#         hx_trigger="change",
#         value="1",
#         _="on change trigger refreshSamples",
#         style="margin-left: 10px; margin-right: 10px;"
#     )
#
# @route('/update_sample_list')
# def get(run_name: str, problem_id: str):
#     """Update sample list when problem is selected"""
#     return Input(
#         type="number",
#         name="sample_id",
#         id="sample-select",
#         placeholder="Sample ID",
#         min="0",
#         hx_trigger="change",
#         value="0",
#         style="margin-left: 10px; margin-right: 10px;"
#     )
#
# @route('/update_run_list')
# def get(run_group: str):
#     """Update run list when run group is selected"""
#     runs = get_available_runs(BASE_LOG_DIR, run_group)
#     return Select(
#         *[Option(run, value=run) for run in runs],
#         name="run_name",
#         id="run-select",
#         hx_get="/update_problem_list",
#         hx_target="#problem-container",
#         hx_trigger="change",
#         style="margin-right: 10px;"
#     )

@rt('/view_results')
def get(run_group: str, run_name: str, problem_id: str, sample_id: str):
    run_group_stats = Details(
        Summary(
            "Run Group Stats [Done]",
            style="cursor: pointer; padding: 10px; background-color: #f0f0f0;",
        ),
        Pre(
            json.dumps(get_run_group_finished_runs(BASE_LOG_DIR, run_group), indent=2),
            style="white-space: pre-wrap; background-color: #f8f8f8; padding: 10px; border-radius: 4px;",
        ),
        style="margin-top: 10px; margin-bottom: 10px;",
    )

    # problem name to problem ids
    problems = get_available_problem_ids(BASE_LOG_DIR, run_group, run_name)

    # TODO buttons for samples
    # nav buttons
    navigation_buttons = Div(
        A(
            "Previous problem",
            href=(
                f"/view_results?"
                    f"run_group={run_group}&"
                    f"run_name={run_name}&"
                    f"problem_id={get_prev_problem_id(problems, int(problem_id))}&"
                    f"sample_id={sample_id}"
            ),
            style=(
                "display:inline-block;"
                "padding:0.5em 1em;"
                "background-color:#007bff;"
                "color:white;"
                "text-decoration:none;"
                "border:none;"
                "border-radius:4px;"
                "font-size:1rem;"
                "cursor:pointer;"
                "transition:background-color 0.2s ease;"
            ),
            id="prev-problem-button",
            onmouseover="this.style.backgroundColor='#0056b3';",
            onmouseout="this.style.backgroundColor='#007bff';",
        ),
        A(
            "Next problem",
            href=(
                f"/view_results?"
                    f"run_group={run_group}&"
                    f"run_name={run_name}&"
                    f"problem_id={get_next_problem_id(problems, int(problem_id))}&"
                    f"sample_id={sample_id}"
            ),
            style=(
                "display:inline-block;"
                "padding:0.5em 1em;"
                "background-color:#007bff;"
                "color:white;"
                "text-decoration:none;"
                "border:none;"
                "border-radius:4px;"
                "font-size:1rem;"
                "cursor:pointer;"
                "transition:background-color 0.2s ease;"
                "margin-left: 5px;"
            ),
            id="next-problem-button",
            onmouseover="this.style.backgroundColor='#0056b3';",
            onmouseout="this.style.backgroundColor='#007bff';",
        ),
        Script("""
            document.addEventListener('keydown', function(e) {
                if (e.key === 'a' || e.key === 'A') {
                    document.getElementById('prev-problem-button').click();
                } else if (e.key === 'd' || e.key === 'D') {
                    document.getElementById('next-problem-button').click();
                }
            });
        """),
        style="display: flex; align-items: center; justify-content: flex-end; margin-bottom: 20px;",
    )

    ui_elements = []
    ui_elements.append(run_group_stats)
    ui_elements.append(navigation_buttons)

    sample_base_dir = os.path.join(BASE_LOG_DIR, run_group, run_name, f"problem_{problem_id}", f"sample_{sample_id}")

    # TODO Navigate even if it doesn't exist, the problems is to constct the database object
    if not os.path.exists(os.path.join(sample_base_dir, "DONE")):
        not_found_content = Div(
            P(f"Result not found for Run Group {run_group} Run Name {run_name} Problem ID {problem_id} Sample ID {sample_id}"),
            style="border: 2px solid red; padding: 10px; border-radius: 4px;"
        )
        ui_elements.append(not_found_content)

        log_path = Path(os.path.join(sample_base_dir, "log.json"))
        if log_path.exists():
            partial_log_content = Div(
                P(
                    Span("Log File Found (Partial):", style="font-weight: bold;"),
                    " Displaying log content below"
                ),
                Details(
                    Summary("View Log Content", style="cursor: pointer; padding: 10px; background-color: #f0f0f0;"),
                    Pre(
                        json.dumps(json.loads(log_path.read_text()), indent=2),
                        style="white-space: pre-wrap; background-color: #f8f8f8; padding: 10px; border-radius: 4px;"
                    ),
                    style="margin-bottom: 20px;"
                )
            )
            ui_elements.append(partial_log_content)


        return ui_elements

    try:


        log_path = os.path.join(sample_base_dir, "log.json")
        config_path = os.path.join(sample_base_dir, "config.json")

        log_data = load_json_data(log_path)
        config_data = load_json_data(config_path)
        max_turns = config_data["max_k"]
        dataset = KernelBenchDataset(
            dataset=dataset_name_to_dataset.get(
                config_data["dataset_name"], "KernelBench/level1"
            )
        )
        problems = dataset.get_problem_ids()

        # TODO properly make this function into a kernelbench eval.py function,
        # so we don't depend on a relative path here
        # fetch timing baselines
        level = int(re.search(r"\d+", config_data["dataset_name"]).group())
        baseline_time_filepath = (
            "../KernelBench/results/timing/H100_tsubame_2/baseline_time_torch.json"
        )
        baseline_torch_compile_time_filepath = (
                "../KernelBench/results/timing/H100_tsubame_2/baseline_time_torch_compile_inductor_default.json"
        )

        baseline_torch_time = fetch_baseline_time_by_problem_id(
            baseline_time_filepath=baseline_time_filepath,
            level=level,
            problem_id=int(problem_id),
        ).get("mean", -1)

        baseline_torch_compile_time = fetch_baseline_time_by_problem_id(
            baseline_time_filepath=baseline_torch_compile_time_filepath,
            level=level,
            problem_id=int(problem_id),
        ).get("mean", -1)

        # add config section at the top
        config_content = Div(
            H2("Summary", style="margin-top: 0; margin-bottom: 10px;"),
            P(
                Span("Run Info:", style="font-weight: bold;"),
                " Run Name: ",
                Code(run_name),
                ", KernelBench Level: ",
                Code(level),
                ", Problem ID: ",
                Code(problem_id),
                ", Sample ID: ",
                Code(sample_id),
            ),
            P(
                "Problem: ",
                Code(os.path.basename(dataset.get_problem_path_by_id(int(problem_id)))),
            ),
            P(
                Span("Strategy:", style="font-weight: bold;"),
                " Max K: ",
                Code(config_data["max_k"]),
                ", Context_strategy: ",
                Code("context-strategy"),
            ),
            Details(
                Summary(
                    "Detailed Configuration",
                    style=("cursor: pointer;padding: 10px;background-color: #f0f0f0;"),
                ),
                Pre(
                    json.dumps(config_data, indent=2),
                    style=(
                        "white-space: pre-wrap;"
                        "background-color: #f8f8f8;"
                        "padding: 10px;"
                        "border-radius: 4px;"
                    ),
                ),
                style="margin-bottom: 20px;",
            ),
            P("Log Path: ", Code(log_path)),
            style=(
                "border-radius: 5px;"
                "padding: 15px;"
                "margin-bottom: 20px;"
                "background-color: #e0ffe0;"
            ),
        )
        ui_elements.append(config_content)

        # TODO probably should get the last run that WORKS (ie has runtime
        # non-negative?)
        # get the final result of the run (i.e. the last execution)
        final_result = log_data.get([*log_data.keys()][-1], None)

        # TODO look into this
        # get performance trajectory
        turn_compile_trajectory, turn_correct_trajectory, turn_runtime_trajectory = (
            get_turn_trajectory_overviews(log_data, max_turns=max_turns)
        )

        # plot the information using plotly
        plot_title = f"Runtime Trajectory for {run_name} - Problem {problem_id} - Sample {sample_id}"

        # BUG: this plot label axes are off! WHY?
        # Create plotly data for runtime trajectory
        runtime_plot_data = Div(
            H2("Runtime Trajectory Plot", style="margin-top: 0; margin-bottom: 10px;"),
            Div(id="runtime-plot"),
            Script(src="https://cdn.plot.ly/plotly-2.32.0.min.js"),
            Script(f"""
                function initializePlot() {{
                    var turns = Array.from({{length: {len(turn_runtime_trajectory)}}}, (_, i) => i + 1);
                    var data = [
                        {{
                            x: turns,
                            y: {turn_runtime_trajectory},
                            type: 'scatter',
                            mode: 'lines+markers',
                            name: 'Runtime'
                        }},
                        {{
                            x: [1, {len(turn_runtime_trajectory)}],
                            y: [{baseline_torch_time}, {baseline_torch_time}],
                            mode: 'lines',
                            name: 'Baseline Torch',
                            line: {{
                                dash: 'dash',
                                color: 'red'
                            }}
                        }},
                        {{
                            x: [1, {len(turn_runtime_trajectory)}],
                            y: [{baseline_torch_compile_time}, {baseline_torch_compile_time}],
                            mode: 'lines',
                            name: 'Baseline Torch Compile',
                            line: {{
                                dash: 'dash',
                                color: 'orange'
                            }}
                        }}
                    ];
                    var layout = {{
                        title: '{plot_title}',
                        xaxis: {{
                            title: 'Turn Number (k)',
                            tickmode: 'linear',
                            automargin: true
                        }},
                        yaxis: {{
                            title: 'Runtime (ms)',
                            automargin: true
                        }},
                        autosize: true
                    }};

                    Plotly.newPlot('runtime-plot', data, layout).then(function() {{
                        Plotly.Plots.resize('runtime-plot');
                    }});
                }}

                // Initialize plot when content loads
                initializePlot();

                // Resize handler
                window.addEventListener('resize', function() {{
                    Plotly.Plots.resize('runtime-plot');
                }});

                // Handle tab visibility changes
                document.addEventListener('visibilitychange', function() {{
                    if (!document.hidden) {{
                        Plotly.Plots.resize('runtime-plot');
                    }}
                }});

                // Additional trigger after a short delay
                setTimeout(initializePlot, 250);
                setTimeout(function() {{
                    Plotly.Plots.resize('runtime-plot');
                }}, 500);
            """),
            style=(
                "border-radius: 5px;"
                "padding: 15px;"
                "margin-bottom: 20px;"
                "background-color: #f0f0ff;"
            ),
        )

        final_result_exists = final_result is not None and "eval_result" in final_result
        if not final_result_exists:
            no_final_result_content = Div(
                Warning("WARNING: No final result found for this run"),
                style="background-color: #fff3cd; color: #856404; padding: 12px; border: 1px solid #ffeeba; border-radius: 4px; margin: 10px 0;"
            )
            ui_elements.append(no_final_result_content)
            final_result = {}

        performance_content = Div(
            H2("Evaluation Results", style="margin-top: 0; margin-bottom: 10px;"),
            Div(
                # Left half - Text content
                Div(
                    P(
                        "Final Sample Stats with turn=: ",
                        Code(str(config_data["max_k"])),
                    ),
                    Span(
                        "Compiled: ",
                        "✅"
                        if final_result_exists
                        and final_result["eval_result"]["compiled"]
                        else "❌",
                        style="margin-right: 20px;",
                    ),
                    Span(
                        "Correctness: ",
                        "✅"
                        if final_result_exists
                        and final_result["eval_result"]["correctness"]
                        else "❌",
                        style="margin-right: 20px;",
                    ),
                    Span(
                        "Runtime: ",
                        Code(
                            final_result["eval_result"]["runtime"]
                            if final_result_exists
                            else "N/A"
                        ),
                        style="margin-right: 20px;",
                    ),
                    Details(
                        Summary(
                            "Runtime Stats",
                            style=(
                                "cursor: pointer;"
                                "padding: 10px;"
                                "background-color: #f0f0f0;"
                            ),
                        ),
                        Pre(
                            json.dumps(final_result, indent=2),
                            style=(
                                "white-space: pre-wrap;"
                                "background-color: #f8f8f8;"
                                "padding: 10px;"
                                "border-radius: 4px;"
                            ),
                        ),
                        style="margin-top: 10px; margin-bottom: 10px;",
                    ),
                    P(Span("Baseline:", style="font-weight: bold;")),
                    Span(
                        "Baseline Torch Time: ",
                        Code(f"{baseline_torch_time} ms"),
                        style="margin-right: 20px;",
                    ),
                    P(),
                    Span(
                        "Baseline Torch Compile Time: ",
                        Code(f"{baseline_torch_compile_time} ms"),
                        style="margin-right: 20px;",
                    ),
                    P(Span("Performance Trajectory:", style="font-weight: bold;")),
                    Table(
                        Tr(
                            Th("Turn", style="font-weight: bold;"),
                            Th("Compiled", style="font-weight: bold;"),
                            Th("Correctness", style="font-weight: bold;"),
                            Th("Runtime", style="font-weight: bold;"),
                        ),
                        *[
                            Tr(
                                Td(str(i + 1)),
                                Td(
                                    "✅"
                                    if turn_compile_trajectory[i]
                                    else "❌"
                                    if turn_compile_trajectory[i] is not None
                                    else "No Data"
                                ),
                                Td(
                                    "✅"
                                    if turn_correct_trajectory[i]
                                    else "❌"
                                    if turn_correct_trajectory[i] is not None
                                    else "No Data"
                                ),
                                Td(str(turn_runtime_trajectory[i])),
                            )
                            for i in range(len(turn_compile_trajectory))
                        ],
                        style=(
                            "width: 100%;"
                            "border-collapse: collapse;"
                            "margin: 10px 0;"
                        ),
                        cellpadding="8",
                        border="1",
                    ),
                    style="flex: 1; padding-right: 15px;",
                ),
                # Right half - Plot
                Div(runtime_plot_data, style="flex: 1;"),
                style="display: flex; gap: 20px;",
            ),
            style=(
                "border-radius: 5px;"
                "padding: 15px;"
                "margin-bottom: 20px;"
                "background-color: #e0e0ff;"
            ),
        )
        ui_elements.append(performance_content)

        # data for each turn
        for turn in range(1, max_turns + 1):
            turn_data = log_data[str(turn)]

            try:
                # create collapsible sections for each turn
                turn_content = Div(
                    H2(
                        f"Turn {turn}",
                        style="margin-top: 0; margin-bottom: 10px;",
                    ),
                    Details(
                        Summary(
                            f"Context - Turn {turn}",
                            style=(
                                "cursor: pointer;"
                                "padding: 10px;"
                                "background-color: #f0f0f0;"
                            ),
                        ),
                        Pre(
                            turn_data["prompt"],
                            style=(
                                "white-space: pre-wrap;"
                                "background-color: #f8f8f8;"
                                "padding: 10px;"
                                "border-radius: 4px;"
                            ),
                        ),
                        style="margin-bottom: 10px;",
                    ),
                    Details(
                        Summary(
                            f"Generated Kernel - Turn {turn}",
                            style=(
                                "cursor: pointer;"
                                "padding: 10px;"
                                "background-color: #f0f0f0;"
                            ),
                        ),
                        Pre(
                            turn_data["kernel_code"],
                            style=(
                                "white-space: pre-wrap;"
                                "background-color: #f8f8f8;"
                                "padding: 10px;"
                                "border-radius: 4px;"
                            ),
                        ),
                        style="margin-bottom: 10px;",
                    ),
                    Details(
                        Summary(
                            f"Evaluation Results - Turn {turn}",
                            style=(
                                "cursor: pointer;"
                                "padding: 10px;"
                                "background-color: #f0f0f0;"
                            ),
                        ),
                        Pre(
                            f"Compiled: {turn_data['eval_result']['compiled']}\n"
                            f"Correct: {turn_data['eval_result']['correctness']}\n"
                            f"Runtime: {turn_data['eval_result']['runtime']} ms\n"
                            f"Metadata: {turn_data['eval_result']['metadata']}",
                            style=(
                                "white-space: pre-wrap;"
                                "background-color: #f8f8f8;"
                                "padding: 10px;"
                                "border-radius: 4px;"
                            ),
                        ),
                        style="margin-bottom: 10px;",
                    ),
                    Details(
                        Summary(
                            f"Feedback - Turn {turn}",
                            style=(
                                "cursor: pointer;"
                                "padding: 10px;"
                                "background-color: #f0f0f0;"
                            ),
                        ),
                        Pre(
                            turn_data.get("feedback", "No feedback available"),
                            style=(
                                "white-space: pre-wrap;"
                                "background-color: #f8f8f8;"
                                "padding: 10px;"
                                "border-radius: 4px;"
                            ),
                        ),
                        style="margin-bottom: 10px;",
                    ),
                    style=(
                        "margin-bottom: 20px;"
                        "border: 1px solid #ddd;"
                        "padding: 10px;"
                        "border-radius: 4px;"
                    ),
                )
            except Exception as e:
                print(e)
                turn_content = P(
                    f"WARNING: Cannot access turn data for turn {turn}; potential Data Corruption",
                    style="color: red;",
                )
            ui_elements.append(turn_content)

        return Div(*ui_elements)

    except Exception as e:
        error_trace = traceback.format_exc()

        return P(
            P(f"Error: {str(e)}", style="color: red;"),
            Details(
                Summary(
                    "View Error Details",
                    style=(
                        "background-color: #ff4444;"
                        "color: white;"
                        "padding: 5px 10px;"
                        "border-radius: 4px;"
                        "cursor: pointer;"
                        "border: none;"
                    ),
                ),
                Pre(
                    error_trace,
                    style=(
                        "white-space: pre-wrap;"
                        "background-color: #f8f8f8;"
                        "padding: 10px;"
                        "border-radius: 4px;"
                    ),
                ),
            ),
        )

# @route('/update_run_info')
# def get(level: str, strategy: str, model: str):
#     """Update run group and name based on mapping selection"""
#     try:
#         mapping = {
#             "level1": RUN_MAPPING_LEVEL_1,
#             "level2": RUN_MAPPING_LEVEL_2,
#             "level3": RUN_MAPPING_LEVEL_3
#         }[level]
#
#         run_info = mapping[strategy][model]
#
#         return Div(
#             Input(
#                 type="hidden",
#                 name="run_group",
#                 value=run_info["run_group"]
#             ),
#             Input(
#                 type="hidden",
#                 name="run_name",
#                 value=run_info["run_name"]
#             ),
#             # Display the selected values
#             P(f"Selected Run Group: {run_info['run_group']}", style="margin: 5px 0;"),
#             P(f"Selected Run Name: {run_info['run_name']}", style="margin: 5px 0;")
#         )
#     except KeyError:
#         return P("No mapping found for this combination", style="color: red;")

# @rt('/')
# def get():
#     return Titled(
#         "KernelBench Multi-Turn Viewer",
#         Script("https://unpkg.com/htmx.org@1.9.10"),
#         Script("https://unpkg.com/hyperscript.org@0.9.12"),
#
#         Div(
#             Span("KernelBench Multi-Turn Viewer"),
#             Span(f"{HOST_IP}:{PORT}", style="margin-left: auto; font-size: 0.6em;"),
#             style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 20px;"
#         ),
#
#         Div(
#             Form(
#                 Div(
#                     # Structured selector
#                     Div(
#                         # Level selector
#                         Select(
#                             # *[Option(f"Level {level[-1]}", value=level) for level in LEVELS],
#                             name="level",
#                             id="level-select",
#                             hx_get="/update_run_info",
#                             hx_include="[name='strategy'],[name='model']",
#                             hx_target="#run-info",
#                             hx_trigger="change",
#                             style="margin-right: 10px; width: 150px;"
#                         ),
#                         # Strategy selector
#                         Select(
#                             *[Option(strategy.replace("_", " ").title(), value=strategy) for strategy in STRATEGIES],
#                             name="strategy",
#                             id="strategy-select",
#                             hx_get="/update_run_info",
#                             hx_include="[name='level'],[name='model']",
#                             hx_target="#run-info",
#                             hx_trigger="change",
#                             style="margin-right: 10px; width: 250px;"
#                         ),
#                         # Model selector
#                         Select(
#                             *[Option(model, value=model) for model in MODELS],
#                             name="model",
#                             id="model-select",
#                             hx_get="/update_run_info",
#                             hx_include="[name='level'],[name='strategy']",
#                             hx_target="#run-info",
#                             hx_trigger="change",
#                             style="margin-right: 10px; width: 200px;"
#                         ),
#                         style="display: flex; align-items: center; margin-bottom: 10px;"
#                     ),
#                     # Container for run info that will be updated
#                     Div(
#                         id="run-info",
#                         style="margin-bottom: 10px;"
#                     ),
#                     # Problem input
#                     Div(
#                         Input(
#                             type="number",
#                             name="problem_id",
#                             placeholder="Problem ID",
#                             min="1",
#                             value="1",
#                             style="margin-right: 10px; width: 150px;"
#                         ),
#                         # Sample input
#                         Input(
#                             type="number",
#                             name="sample_id",
#                             placeholder="Sample ID",
#                             min="0",
#                             value="0",
#                             style="margin-right: 10px; width: 150px;"
#                         ),
#                         style="display: flex; align-items: center; margin-bottom: 10px;"
#                     ),
#                     # Use subset checkbox
#                     Div(
#                         Label(
#                             Input(
#                                 type="checkbox",
#                                 name="use_subset",
#                                 value="true",
#                                 style="margin-right: 5px;"
#                             ),
#                             "Use Subset",
#                             style="display: flex; align-items: center;"
#                         ),
#                         style="margin-bottom: 10px;"
#                     ),
#                     Button(
#                         "View Results",
#                         type="submit",
#                         style="background-color: green; color: white; width: 200px;"
#                     ),
#                 ),
#                 hx_post="/view_results",
#                 hx_target="#result"
#             ),
#             Div(id="result", style="width: 100%;"),
#             style="width: 100%;"
#         )
#     )

serve(port=PORT)
