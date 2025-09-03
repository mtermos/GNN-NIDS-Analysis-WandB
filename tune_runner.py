#!/usr/bin/env python3
"""
Simple Hyperparameter Tuning Runner with Optuna

This script uses Optuna to optimize hyperparameters by creating custom CONFIG 
instances and replacing the CONFIG import in main.py to run trials.

Just modify the CONFIG section below and run:
    python simple_tune_runner.py
"""

import os
import json
import optuna
import random
import subprocess
import tempfile
from pathlib import Path
import sys
from datetime import datetime

# =============================================================================
# CONFIG - Modify these settings
# =============================================================================

# Models to test (must match keys in main.py all_models dict)
# Available models: "e_graphsage", "e_graphsage_no_sampling", "e_gat_no_sampling", "e_gat_sampling"
# MODELS_TO_TEST = ["e_graphsage"]
# MODELS_TO_TEST = ["e_graphsage", "e_graphsage_no_sampling", "e_gat_no_sampling", "e_gat_sampling"]
# MODELS_TO_TEST = ["e_gat_no_sampling", "e_gat_sampling"]
MODELS_TO_TEST = ["e_graphsage", "e_gat"]
NEIGHBOR_SAMPLING = False

# Number of trials
N_TRIALS = 1

# Enable WandB logging (set to False if you have authentication issues)
USE_WANDB = False

# Hyperparameter choices (all as lists)
HP_RANGES = {
    "learning_rate": [0.001, 0.002, 0.005, 0.01],
    "weight_decay": [0.0001, 0.001, 0.01],
    # "num_layers": [1, 2, 3],
    # "num_layers": [1, 2],
    "num_layers": [1],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "residual": [True, False],
    "aggregation": ["mean", "sum", "max"],
    # "loss_name": ["vanilla_ce", "focal", "ldam_drw"],
    "focal_gamma": [1.0, 1.5, 2.0, 2.5, 3.0],
}

# Fixed hyperparameters
FIXED_HPS = {
    "dataset_name": "cic_ids_2017_5_percent",
    "original_path": "testing_dfs/cic_ids_2017_5_percent.parquet",
    "max_epochs": 5,
    "selected_models": MODELS_TO_TEST,
    "focal_alpha": 0.25,
    "loss_name": "ldam_drw",
    "edge_update": True,
}

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
    
    # Sample from choices using choice method
    for param, choices in HP_RANGES.items():
        hps[param] = choice(choices, param)
    
    if hps["num_layers"] == 1:
        # hps["ndim_out"] = [128]
        ndim_out_str = choice(["64", "128", "256"], "ndim_out_1layer")
        hps["ndim_out"] = [int(ndim_out_str)]
        if NEIGHBOR_SAMPLING:
            hps["number_neighbors"] = [25]
        else:
            hps["number_neighbors"] = None
    elif hps["num_layers"] == 2:
        # hps["ndim_out"] = [128,128]
        ndim_out_str = choice(["64,64", "128,128", "256,256"], "ndim_out_2layer")
        hps["ndim_out"] = [int(x) for x in ndim_out_str.split(",")]
        if NEIGHBOR_SAMPLING:
            hps["number_neighbors"] = [25, 10]
        else:
            hps["number_neighbors"] = None
    else:  # 3 layers   
        # hps["ndim_out"] = [128,128,128]
        ndim_out_str = choice(["64,64,64", "128,128,64", "128,128,128"], "ndim_out_3layer")
        hps["ndim_out"] = [int(x) for x in ndim_out_str.split(",")]   
        if NEIGHBOR_SAMPLING:
            hps["number_neighbors"] = [25, 10, 10]
        else:
            hps["number_neighbors"] = None

    return hps

def create_custom_config(hps):
    """Create a custom CONFIG instance with the desired hyperparameters"""
    config_code = f"""
# Custom CONFIG for this trial
from src.config import Config

CONFIG = Config(
    # Dataset configuration
    dataset_name="{hps['dataset_name']}",
    original_path="{hps['original_path']}",
    
    # Model architecture
    ndim_out={hps['ndim_out']},
    num_layers={hps['num_layers']},
    number_neighbors={hps['number_neighbors']},
    dropout={hps['dropout']},
    residual={hps['residual']},
    aggregation="{hps['aggregation']}",
    edge_update={hps['edge_update']},
    selected_models={hps['selected_models']},

    # Training configuration
    using_wandb={USE_WANDB},
    max_epochs={hps['max_epochs']},
    learning_rate={hps['learning_rate']},
    weight_decay={hps['weight_decay']},
    
    # Loss function configuration
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
    """Run a single trial"""
    print(f"\n=== Trial {trial_num}/{N_TRIALS} ===")
    print(f"Hyperparams: {hps}")
    
    with tempfile.TemporaryDirectory() as td:
        # Copy the entire project structure to temp directory
        import shutil
        project_files = ["src", "testing_dfs", "local_variables.py", "requirements.txt"]
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
        
        # Run the trial
        try:
            result = subprocess.run(
                [sys.executable, str(patched_file)],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
                cwd=td
            )
            
            if result.returncode == 0:
                print("✓ Trial completed successfully")
                return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
            else:
                print(f"✗ Trial failed: {result.stderr}")
                return {"success": False, "stdout": result.stdout, "stderr": result.stderr}
                
        except subprocess.TimeoutExpired:
            print("✗ Trial timed out")
            return {"success": False, "error": "Timeout"}

def main():
    """Main function"""
    print("Simple Hyperparameter Tuning Runner (with Optuna)")
    print("=" * 50)
    print(f"Models: {MODELS_TO_TEST}")
    print(f"Trials: {N_TRIALS}")
    print(f"WandB: {USE_WANDB}")
    
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
    results_file = os.path.join("tuning_results", f"tuning_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print("Tuning completed!")

if __name__ == "__main__":
    main()
