# On Time, Within Budget: Constraint-Driven Online Resource Allocation for Agentic Workflows
## Paper Status

This repository is a compact NeurIPS submission backup for the main experimental pipeline. It contains the code needed to construct rollout-derived DAG pools and run the main simulation experiments. Generated experimental results, plotting scripts, analysis outputs, local caches, and machine-specific paths are intentionally excluded.

## Project Structure

    loader_neurips_submission_main_pre_rollout/
    ├── simulator/
    │   ├── simulator.py                         # Core execution simulator
    │   ├── run_experiment.py                    # Policy construction and experiment runner
    │   ├── mc_portfolio_policy.py               # MC portfolio rollout policy
    │   ├── LOADER_policies.py                   # Baseline scheduling policies
    │   ├── deadline_policies.py                 # Deadline-aware baselines
    │   ├── dag.py                               # DAG and node definitions
    │   └── types.py                             # Shared data types
    ├── codeflow/
    │   ├── data/                                # Pre-rollout CodeFlow task datasets
    │   └── run/                                 # CodeFlow rollout generation utilities
    ├── ProofFlow/
    │   ├── data/                                # Pre-rollout ProofFlow benchmark JSON
    │   ├── source_pickles/                      # Source artifacts for ProofFlow node rerolling
    │   ├── prompts/                             # ProofFlow prompts
    │   ├── ProofFlow/                           # ProofFlow package code
    │   ├── run_proofflow_node_rollouts.py       # ProofFlow rollout generation script
    │   └── convert_proofflow_pickle_to_dag_pool.py
    ├── build_loader_dag_pool_from_rollouts.py   # Converts rollout records to DAG pools
    ├── build_aligned_union_dag_pools.py         # Builds aligned multi-model DAG pools
    ├── run_experiment.sh                        # Main experiment driver
    ├── gen_jobs.py                              # Generates distributed shard jobs
    ├── run_shard.py                             # Runs one experiment shard
    ├── merge_shards.py                          # Merges shard outputs
    └── requirements.txt                         # Python dependencies

## Interactive Demo

The CodeFlow budget/deadline prediction demo is available as a static GitHub
Pages artifact under `docs/codeflow-prediction/`. It uses sanitized aggregate
matrices only and does not include raw worker logs or private run metadata.

## Installation

1. Create a Python environment. Python 3.10+ is recommended.

2. Install dependencies:

        pip install -r requirements.txt

Key dependencies include: `numpy`, `pandas`, `scipy`, `tqdm`, `matplotlib`, `numba`, and `pytest`.

Additional optional dependencies for rollout generation are listed in `codeflow/requirements.txt` and `ProofFlow/requirements.txt`.

## Usage

### Configuration

Before running the main experiment, prepare a DAG pool from rollout data and configure the following environment variables:

* `DAG_POOL_PATH`: Path to the input DAG pool JSONL file.
* `RUN_NAME`: Name of the experiment run.
* `N_DAGS`: Number of DAGs to include.
* `N_EXEC_TRIALS`: Number of execution trials per setting.
* `BUDGET_DEADLINE_VALUES_MAP`: Budget and deadline grid, e.g. `0.03:60,300;1:60,300`.
* `USE_DAG_INDICES_FILE`: Optional fixed subset index file.

GPU keepalive is disabled by default in this backup. If your platform requires it, set:

    export GPU_SCRIPT=/path/to/gpu_keepalive.py

### Running the Main Experiment

The `run_experiment.sh` script executes the main distributed simulation workflow.

    export DAG_POOL_PATH=/path/to/dag_pool.jsonl
    export RUN_NAME=my_main_run
    export N_DAGS=1000
    export N_EXEC_TRIALS=100
    export BUDGET_DEADLINE_VALUES_MAP='0.03:60,300;1:60,300'

    bash run_experiment.sh prepare
    bash run_experiment.sh auto
    bash run_experiment.sh status
    bash run_experiment.sh merge

For a single local shard:

    bash run_experiment.sh shard 0

## Pipeline Steps Explained

1. Rollout Collection: CodeFlow or ProofFlow task data is used to collect per-node rollout statistics before the main simulation.

2. DAG Pool Construction: Rollout records are converted into DAG pools using `build_loader_dag_pool_from_rollouts.py` or `ProofFlow/convert_proofflow_pickle_to_dag_pool.py`.

3. Multi-Model Alignment: If multiple model-specific rollout pools are available, `build_aligned_union_dag_pools.py` constructs an aligned multi-model DAG pool for portfolio experiments.

4. Main Simulation: `run_experiment.sh` generates shard jobs and runs policies such as `mc_portfolio_rollout`, `uniform`, `sequential`, `random`, and static baselines over the configured budget/deadline grid.

5. Merge: `merge_shards.py` combines shard-level raw outputs into a merged experiment directory.

## Data Format

Pre-rollout datasets included in this backup:

* `codeflow/data/codeflowbench_comp_test.json`
* `codeflow/data/codeflowbench_repo.json`
* `ProofFlow/data/benchmark_0409.json`
* `ProofFlow/source_pickles/*.pickle`, used as source artifacts for ProofFlow node rerolling

DAG pool files are JSONL files where each line represents one DAG. Each DAG contains:

* Node topology, including parents, children, and layer information.
* Per-node latency, success, cost, and token-cost statistics.
* Empirical rollout samples used by the simulator.
* Optional per-model templates and empirical samples for multi-model portfolio experiments.

Selected index files are JSON files with the following format:

    {"dag_indices": [0, 1, 2]}

## Notes

This submission backup does not include generated results, analysis scripts, plotting scripts, test-only files, local session logs, virtual environments, Git metadata, or user-specific filesystem paths.
