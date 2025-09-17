#!/usr/bin/env python3
"""
Simple Hyperparameter Tuning Runner with Optuna - Fixed Version

This script uses Optuna to optimize hyperparameters by creating custom CONFIG 
instances and replacing the CONFIG import in main.py to run trials.

This version fixes subprocess timeout issues by implementing better process management,
proper cleanup, and more robust timeout handling.

Just modify the CONFIG section below and run:
    python tune_runner_fixed.py
"""

import os
import json
import optuna
import random
import subprocess
import tempfile
import signal
import time
import psutil
from pathlib import Path
import sys
from datetime import datetime
import threading
import queue

from local_variables import local_datasets_path

# =============================================================================
# CONFIG - Modify these settings
# =============================================================================

# Models to test (must match keys in main.py all_models dict)
# Available models: "e_graphsage", "e_graphsage_no_sampling", "e_gat_no_sampling", "e_gat_sampling"
# MODELS_TO_TEST = ["e_graphsage"]
# MODELS_TO_TEST = ["e_graphsage", "e_graphsage_no_sampling", "e_gat_no_sampling", "e_gat_sampling"]
# MODELS_TO_TEST = ["e_gat_no_sampling", "e_gat_sampling"]
MODELS_TO_TEST = ["e_graphsage", "e_gat"]
NEIGHBOR_SAMPLING = [True, False]

# DATASET_NAME = "cic_ids_2017"
DATASET_NAME = "cic_ton_iot"
# DATASET_NAME = "cic_bot_iot"
# DATASET_NAME = "cic_ton_iot_modified"
# DATASET_NAME = "nf_ton_iotv2_modified"
# DATASET_NAME = "ccd_inid_modified"
# DATASET_NAME = "nf_uq_nids_modified"
# DATASET_NAME = "edge_iiot"
# DATASET_NAME = "nf_cse_cic_ids2018"
# DATASET_NAME = "nf_bot_iotv2"
# DATASET_NAME = "nf_uq_nids"
# DATASET_NAME = "x_iiot"

# Number of trials
N_TRIALS = 20

# Enable WandB logging (set to False if you have authentication issues)
USE_WANDB = True

# Timeout settings (in seconds)
TRIAL_TIMEOUT = 1800  # 30 minutes per trial
PROCESS_CLEANUP_TIMEOUT = 10  # 10 seconds to cleanup processes

# Hyperparameter choices (all as lists)
HP_RANGES = {
    "learning_rate": [0.001, 0.0025, 0.005, 0.0075, 0.01],
    "weight_decay": [0.0001, 0.001, 0.01],
    # "num_layers": [1, 2, 3],
    "num_layers": [1, 2],
    # "num_layers": [1],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    # "residual": [True, False],
    "aggregation": ["mean", "sum"],
    "loss_name": ["vanilla_ce", "focal", "ldam_drw"],
    "focal_gamma": [1.0, 1.5, 2.0, 2.5, 3.0],
}

# Fixed hyperparameters
FIXED_HPS = {
    "dataset_name": DATASET_NAME,
    "original_path": os.path.join(local_datasets_path, DATASET_NAME, f"{DATASET_NAME}.parquet"),
    "max_epochs": 500,
    "early_stopping_patience": 20,
    # "dataset_name": "cic_ids_2017_5_percent",
    # "original_path": "testing_dfs/cic_ids_2017_5_percent.parquet",
    # "max_epochs": 5,
    "selected_models": MODELS_TO_TEST,
    # "focal_alpha": 0.25,
    "focal_alpha": "weighted_class_counts",
    # "loss_name": "ldam_drw",
    "edge_update": False,
    "residual": True,
}

# =============================================================================
# PROCESS MANAGEMENT UTILITIES
# =============================================================================


def kill_process_tree(pid, timeout=PROCESS_CLEANUP_TIMEOUT):
    """Kill a process and all its children recursively"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # First try to terminate gracefully
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait for processes to terminate
        gone, still_alive = psutil.wait_procs(children, timeout=timeout)

        # Force kill any remaining processes
        for child in still_alive:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Finally kill the parent
        try:
            parent.terminate()
            parent.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def monitor_process(process, timeout, result_queue):
    """Monitor a process and put result in queue when done or timeout"""
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        result_queue.put({
            'success': True,
            'returncode': process.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'timeout': False
        })
    except subprocess.TimeoutExpired:
        result_queue.put({
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': '',
            'timeout': True
        })
        # Force kill the process
        kill_process_tree(process.pid)

# =============================================================================
# MAIN CODE
# =============================================================================


def sample_hyperparameters(trial):

    def choice(opts, name=None):
        if trial is None:
            return random.choice(opts)
        return trial.suggest_categorical(name or f"choice_{random.randint(0,999999)}", opts)

    """Sample hyperparameters for a trial using Optuna"""
    hps = FIXED_HPS.copy()

    use_neighbor_sampling = choice(NEIGHBOR_SAMPLING, "use_neighbor_sampling")
    # Sample from choices using choice method
    for param, choices in HP_RANGES.items():
        hps[param] = choice(choices, param)

    if hps["num_layers"] == 1:
        # hps["ndim_out"] = [128]
        ndim_out_str = choice(["64", "128", "256"], "ndim_out_1layer")
        hps["ndim_out"] = [int(ndim_out_str)]
        if use_neighbor_sampling:
            hps["number_neighbors"] = [25]
        else:
            hps["number_neighbors"] = None
    elif hps["num_layers"] == 2:
        # hps["ndim_out"] = [128,128]
        ndim_out_str = choice(
            ["64,64", "128,128", "256,256"], "ndim_out_2layer")
        hps["ndim_out"] = [int(x) for x in ndim_out_str.split(",")]
        if use_neighbor_sampling:
            hps["number_neighbors"] = [25, 10]
        else:
            hps["number_neighbors"] = None
    else:  # 3 layers
        # hps["ndim_out"] = [128,128,128]
        ndim_out_str = choice(
            ["64,64,64", "128,128,64", "128,128,128"], "ndim_out_3layer")
        hps["ndim_out"] = [int(x) for x in ndim_out_str.split(",")]
        if use_neighbor_sampling:
            hps["number_neighbors"] = [25, 10, 10]
        else:
            hps["number_neighbors"] = None

    return hps


def create_custom_config(hps):
    """Create a custom CONFIG instance with the desired hyperparameters"""

    path_lit = json.dumps(str(hps["original_path"]))
    #
    config_code = f"""
# Custom CONFIG for this trial
from src.config import Config

CONFIG = Config(
    dataset_name="{hps['dataset_name']}",
    original_path={path_lit},

    ndim_out={hps['ndim_out']},
    num_layers={hps['num_layers']},
    number_neighbors={hps['number_neighbors']},
    dropout={hps['dropout']},
    residual={hps['residual']},
    aggregation="{hps['aggregation']}",
    edge_update={hps['edge_update']},
    selected_models={hps['selected_models']},

    using_wandb={USE_WANDB},
    max_epochs={hps['max_epochs']},
    learning_rate={hps['learning_rate']},
    weight_decay={hps['weight_decay']},
    
    loss_name="{hps['loss_name']}",
    focal_alpha={repr(hps['focal_alpha'])},
    focal_gamma={hps['focal_gamma']},
)
"""
    return config_code


def patch_main_file(hps):
    """Patch main.py with new hyperparameters and models"""
    with open("main.py", 'r') as f:
        src = f.read()

    # Create custom CONFIG
    custom_config = create_custom_config(hps)

    # Replace the CONFIG import with our custom one
    old_import = 'from src.config import CONFIG'
    new_import = custom_config
    src = src.replace(old_import, new_import)

    return src


def run_trial(hps, trial_num):
    """Run a single trial with improved process management"""
    print(f"\n=== Trial {trial_num}/{N_TRIALS} ===")
    print(f"Hyperparams: {hps}")

    with tempfile.TemporaryDirectory() as td:
        # Copy the entire project structure to temp directory
        import shutil
        project_files = ["src", "testing_dfs",
                         "local_variables.py", "requirements.txt"]
        for item in project_files:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.copytree(item, Path(td) / item)
                else:
                    shutil.copy2(item, td)

        # Patch and write main.py
        patched_content = patch_main_file(hps)
        patched_file = Path(td) / "main.py"
        with open(patched_file, 'w') as f:
            f.write(patched_content)

        # Run the trial with improved process management
        try:
            print(f"Starting trial {trial_num} (timeout: {TRIAL_TIMEOUT}s)")

            # Start the process
            process = subprocess.Popen(
                [sys.executable, str(patched_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=td
            )

            # Use a queue to get results from the monitoring thread
            result_queue = queue.Queue()
            monitor_thread = threading.Thread(
                target=monitor_process,
                args=(process, TRIAL_TIMEOUT, result_queue)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # Wait for the result
            try:
                result = result_queue.get(
                    timeout=TRIAL_TIMEOUT + 30)  # Extra buffer
            except queue.Empty:
                # This shouldn't happen, but just in case
                print("✗ Trial monitoring timed out")
                kill_process_tree(process.pid)
                return {"success": False, "error": "Monitoring timeout"}

            # Wait for monitor thread to finish
            monitor_thread.join(timeout=5)

            if result["timeout"]:
                print("✗ Trial timed out - process killed")
                return {"success": False, "error": "Timeout", "stdout": result["stdout"], "stderr": result["stderr"]}
            elif result["returncode"] == 0:
                print("✓ Trial completed successfully")
                return {"success": True, "stdout": result["stdout"], "stderr": result["stderr"]}
            else:
                print(
                    f"✗ Trial failed with return code {result['returncode']}")
                return {"success": False, "stdout": result["stdout"], "stderr": result["stderr"]}

        except Exception as e:
            print(f"✗ Trial failed with exception: {e}")
            # Ensure process is killed
            try:
                if 'process' in locals():
                    kill_process_tree(process.pid)
            except:
                pass
            return {"success": False, "error": str(e)}


def main():
    """Main function"""
    print("Simple Hyperparameter Tuning Runner (with Optuna) - Fixed Version")
    print("=" * 70)
    print(f"Models: {MODELS_TO_TEST}")
    print(f"Trials: {N_TRIALS}")
    print(f"WandB: {USE_WANDB}")
    print(f"Trial Timeout: {TRIAL_TIMEOUT}s")
    print(f"Process Cleanup Timeout: {PROCESS_CLEANUP_TIMEOUT}s")

    study = optuna.create_study(direction="maximize")
    results = []

    for i in range(N_TRIALS):
        trial = study.ask()
        hps = sample_hyperparameters(trial)
        result = run_trial(hps, i + 1)
        results.append({"trial": i + 1, "hps": hps, "result": result})

        if result["success"]:
            # Simple score extraction (you may need to adjust this)
            score = 0.5  # Placeholder - extract actual score from stdout
            study.tell(trial, score)
        else:
            study.tell(trial, 0.0)  # Failed trials get score 0

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("tuning_results", exist_ok=True)
    results_file = os.path.join(
        "tuning_results", f"tuning_results_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    print("Tuning completed!")


if __name__ == "__main__":
    main()
