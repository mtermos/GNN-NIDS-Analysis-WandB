#!/usr/bin/env python3
"""
Hyperparameter tuning for adap_gnn_v3.py without modifying the original file.

- Copies adap_gnn_v3.py to a temp dir per trial, patches only:
  * top-level hyperparameters (learning_rate, weight_decay, num_layers, ndim_out, number_neighbors, dropout, residual, aggregation)
  * flags: USE_MULTI_SEED_EVAL=False, using_wandb=<cli>
  * empty models_dict to include desired model(s) dynamically
  * replaces the original bottom-run block with a TUNER export that writes JSON
- Executes the patched copy with 'python patched_adap_gnn_v3.py' and reads JSON results.
- Simple Unicode handling for Windows compatibility

USAGE:
  python tune_adap_gnn_py.py --py adap_gnn_v3.py --model e_gat --n-trials 10 --out outputs/egat.jsonl --wandb
  python tune_adap_gnn_py.py --py adap_gnn_v3.py --model e_graphsage --n-trials 10 --optuna
  python tune_adap_gnn_py.py --py adap_gnn_v3.py --model both --n-trials 12
  python tune_adap_gnn_py.py --py adap_gnn_v3.py --model both --n-trials 2 --out outputs/both.jsonl
"""

import argparse, os, json, random, math, re, subprocess, tempfile
from pathlib import Path
import sys

# Windows-specific encoding setup
if os.name == 'nt':
    import locale
    import codecs
    
    # Force UTF-8 encoding for Windows
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
    
    # Try to set locale to UTF-8
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except:
            pass
    
    # Ensure stdout/stderr use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    
    print("Windows encoding setup completed:")
    print(f"  PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'Not set')}")
    print(f"  PYTHONUTF8: {os.environ.get('PYTHONUTF8', 'Not set')}")
    print(f"  PYTHONLEGACYWINDOWSSTDIO: {os.environ.get('PYTHONLEGACYWINDOWSSTDIO', 'Not set')}")
    print(f"  Default encoding: {sys.getdefaultencoding()}")
    print(f"  File system encoding: {sys.getfilesystemencoding()}")

import argparse, os, json, random, math, re, subprocess, tempfile
from pathlib import Path
import sys

# === DYNAMIC GNN MODEL CONFIGURATION ===
# Add new GNN variants here by extending this dictionary
GNN_MODELS = {
    "e_gat": {
        "class_name": "EGAT",
        "constructor_params": {
            "ndim_in": "ndim",
            "edim": "edim", 
            "ndim_out": "ndim_out",
            "num_layers": "num_layers",
            "activation": "activation",
            "dropout": "dropout",
            "residual": "residual",
            "num_class": "len(labels_mapping)",
            "num_neighbors": "None"
        },
        "model_key_template": "e_gat_no_sampling",
        "description": "Edge-aware Graph Attention Network without sampling"
    },
    "e_graphsage": {
        "class_name": "EGRAPHSAGE",
        "constructor_params": {
            "ndim_in": "ndim",
            "edim": "edim",
            "ndim_out": "ndim_out", 
            "num_layers": "num_layers",
            "activation": "activation",
            "dropout": "dropout",
            "residual": "residual",
            "num_class": "len(labels_mapping)",
            "num_neighbors": "number_neighbors",
            "aggregation": "aggregation"
        },
        "model_key_template": "e_graphsage_{aggregation}",
        "description": "Edge-aware GraphSAGE with configurable aggregation"
    },
    "e_graphsage_no_sampling": {
        "class_name": "EGRAPHSAGE",
        "constructor_params": {
            "ndim_in": "ndim",
            "edim": "edim",
            "ndim_out": "ndim_out", 
            "num_layers": "num_layers",
            "activation": "activation",
            "dropout": "dropout",
            "residual": "residual",
            "num_class": "len(labels_mapping)",
            "num_neighbors": "None",
            "aggregation": "aggregation"
        },
        "model_key_template": "e_graphsage_{aggregation}_no_sampling",
        "description": "Edge-aware GraphSAGE with configurable aggregation and no sampling"
    },
    # Add new GNN variants here following the same pattern:
    # "new_gnn_variant": {
    #     "class_name": "NEWGNNCLASS",
    #     "constructor_params": {
    #         "param1": "value1",
    #         "param2": "value2",
    #         # ... other parameters
    #     },
    #     "model_key_template": "new_gnn_{param1}",
    #     "description": "Description of the new GNN variant"
    # }
}

def generate_model_dict(models_to_include, hps):
    """
    Dynamically generate the models_dict based on which models to include.
    
    Args:
        models_to_include: List of model names to include (e.g., ["e_gat", "e_graphsage"])
        hps: Hyperparameters dictionary
    
    Returns:
        String representation of the models_dict
    """
    if not models_to_include:
        raise ValueError("At least one model must be specified")
    
    model_entries = []
    
    for model_name in models_to_include:
        if model_name not in GNN_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(GNN_MODELS.keys())}")
        
        model_config = GNN_MODELS[model_name]
        class_name = model_config["class_name"]
        params = model_config["constructor_params"]
        
        # Generate constructor call
        param_list = []
        for param_name, param_value in params.items():
            param_list.append(f"{param_name}={param_value}")
        
        constructor_call = f"{class_name}({', '.join(param_list)})"
        
        # Generate model key
        model_key = model_config["model_key_template"]
        if "aggregation" in model_key:
            model_key = f"'{model_key.format(aggregation=hps['aggregation'])}'"
        else:
            model_key = f"'{model_key}'"
        
        model_entries.append(f"    {model_key}: {constructor_call}")
    
    models_dict = "models_dict = {\n" + ",\n".join(model_entries) + "\n}"
    print(f"==>> models_dict: {models_dict}")
    return models_dict

def get_available_models():
    """Get list of available model names for CLI help."""
    return list(GNN_MODELS.keys()) + ["both"]

def test_encoding_setup():
    """Test and display current encoding configuration."""
    print("=== Encoding Configuration Test ===")
    print(f"Platform: {os.name}")
    print(f"Python version: {sys.version}")
    print(f"Default encoding: {sys.getdefaultencoding()}")
    print(f"File system encoding: {sys.getfilesystemencoding()}")
    print(f"Locale encoding: {locale.getpreferredencoding() if 'locale' in globals() else 'N/A'}")
    print(f"Environment variables:")
    for key in ['PYTHONIOENCODING', 'PYTHONUTF8', 'PYTHONLEGACYWINDOWSSTDIO']:
        print(f"  {key}: {os.environ.get(key, 'Not set')}")
    print("=" * 40)
    
    # Test file operations
    print("\n=== File Operations Test ===")
    test_content = "Hello World! 🌍 测试"
    test_file = "test_encoding_temp.txt"
    
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        print("✓ UTF-8 write successful")
        
        with open(test_file, "r", encoding="utf-8") as f:
            read_content = f.read()
        print("✓ UTF-8 read successful")
        
        if read_content == test_content:
            print("✓ Content matches exactly")
        else:
            print("✗ Content mismatch")
            
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
    
    print("=" * 40)

def sample_hparams(trial=None):
    def choice(opts, name=None):
        if trial is None:
            return random.choice(opts)
        return trial.suggest_categorical(name or f"choice_{random.randint(0,999999)}", opts)

    hps = {}

    # hps["dataset"] = "testing_dfs/cic_ids_2017_5_percent.parquet"

    # hps["dataset_name"] = "cic_ids_2017_5_percent"
    # hps["original_path"] = "/Users/mortadatermos/Desktop/phd/test code/testing_dfs/cic_ids_2017_5_percent.parquet"
    # hps["max_epochs"] = 4

    hps["dataset_name"] = "cic_ids_2017"
    hps["original_path"] = "C:/Users/Administrateur/Desktop/datasets/cic_ids_2017/cic_ids_2017.parquet"
    hps["max_epochs"] = 500

    # Use clean, predefined hyperparameter choices
    hps["learning_rate"] = 0.005
    # hps["learning_rate"] = choice([0.001, 0.0025, 0.005, 0.0075, 0.01], "learning_rate")

    hps["weight_decay"] = 0.001
    # hps["weight_decay"] = choice([0.000001, 0.00001, 0.0001, 0.001, 0.01], "weight_decay")

    hps["num_layers"] = 2
    # hps["num_layers"] = choice([1, 2, 3], "num_layers")
    
    # Fixed dimension options based on number of layers
    if hps["num_layers"] == 1:
        hps["ndim_out"] = [128]
        # hps["ndim_out"] = choice([[64], [128], [256]], "ndim_out_1layer")
    elif hps["num_layers"] == 2:
        hps["ndim_out"] = [128,128]
        # hps["ndim_out"] = choice([[64,64], [128,128], [256,256]], "ndim_out_2layer")
    else:  # 3 layers
        hps["ndim_out"] = [128,128,128]
        # hps["ndim_out"] = choice([[64,64,64], [128,128,64], [128,128,128]], "ndim_out_3layer")
    
    hps["dropout"] = 0.4
    # hps["dropout"] = choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "dropout")
    hps["residual"] = True
    # hps["residual"] = choice([True, False], "residual")
    
    if hps["num_layers"] == 1:
        hps["number_neighbors"] = [25]
        # hps["number_neighbors"] = choice([[5], [10], [15], [25], [50]], "number_neighbors_1layer")
    elif hps["num_layers"] == 2:
        hps["number_neighbors"] = [25, 10]
        # hps["number_neighbors"] = choice([[5, 5], [10, 5], [15,10], [25, 10], [50, 25]], "number_neighbors_2layer")
    else:  # 3 layers
        hps["number_neighbors"] = [25, 10, 10]
        # hps["number_neighbors"] = choice([[5, 5, 5], [10, 5, 5], [15,10, 10], [25, 10, 10], [50, 25, 10]], "number_neighbors_3layer")
    # Fixed neighbor options
    # hps["number_neighbors"] = choice([[5,5], [10,5], [15,10], [25,10], [50,25]], "number_neighbors")
    
    hps["aggregation"] = "mean"
    # hps["aggregation"] = choice(["mean", "sum"], "aggregation")
    # hps["aggregation"] = choice(["mean", "sum", "pool"], "aggregation")


    # hps["loss_name"] = "vanilla_ce"
    # hps["loss_name"] = "ce_cb"
    hps["loss_name"] = "focal"
    # hps["loss_name"] = "ldam_drw"
    # hps["loss_name"] = "logit_adj"
    # hps["loss_name"] = "balanced_softmax"
    # hps["loss_name"] = choice(["vanilla_ce", "ce_cb", "focal", "ldam_drw", "logit_adj", "balanced_softmax"], "loss_name")
    # hps["loss_name"] = choice(["vanilla_ce", "ce_cb", "focal", "logit_adj", "balanced_softmax"], "loss_name")
    hps["focal_gamma"] = 2.0
    hps["focal_alpha"] = "weighted_class_counts"
    hps["class_counts_scheme"] = "effective"
    hps["class_counts_beta"] = 0.999
    hps["class_counts_normalize"] = "max1"
    hps["cb_beta"] = 0.999
    hps["ldam_C_margin"] = 0.5
    hps["drw_start"] = 10
    hps["cb_beta_drw"] = 0.999
    hps["logit_adj_tau"] = 1.0
    # hps["focal_gamma"] = choice([1.0, 2.0, 3.0], "focal_gamma")
    # hps["focal_alpha"] = choice([None, 0.25, 0.5, 0.75], "focal_alpha")
    # hps["cb_beta"] = choice([0.999, 0.9999, 0.99999], "cb_beta")
    # hps["ldam_C_margin"] = choice([0.25, 0.5, 0.75], "ldam_C_margin")
    # hps["drw_start"] = choice([5, 10, 15], "drw_start")
    # hps["cb_beta_drw"] = choice([0.999, 0.9999, 0.99999], "cb_beta_drw")
    # hps["logit_adj_tau"] = choice([1.0, 2.0, 3.0], "logit_adj_tau")

    hps["multi_seed"] = "False"
    return hps

def patch_python(orig_path: Path, tmp_dir: Path, hps: dict, target_model: str, use_wandb: bool) -> Path:
    try:
        src = Path(orig_path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print("Warning: UTF-8 decode failed, trying with latin-1 encoding")
        try:
            src = Path(orig_path).read_text(encoding="latin-1")
        except Exception as e:
            print(f"Warning: Latin-1 decode also failed: {e}")
            # Try with system default encoding
            src = Path(orig_path).read_text(encoding=None, errors='replace')
    
    # Simple string replacements - much safer than regex
    replacements = [
        ('learning_rate = 0.005', f'learning_rate = {hps["learning_rate"]}'),
        ('weight_decay = 0.01', f'weight_decay = {hps["weight_decay"]}'),
        ('ndim_out = [128, 128]', f'ndim_out = {hps["ndim_out"]}'),
        ('num_layers = 2', f'num_layers = {hps["num_layers"]}'),
        ('number_neighbors = [25, 10]', f'number_neighbors = {hps["number_neighbors"]}'),
        ('dropout = 0.5', f'dropout = {hps["dropout"]}'),
        ('residual = True', f'residual = {str(hps["residual"])}'),
        ('aggregation = "mean"', f'aggregation = "{hps["aggregation"]}"'),
        ('using_wandb = True', f'using_wandb = {str(bool(use_wandb))}'),
        ('original_path = "testing_dfs\cic_ids_2017_5_percent.parquet"', f'original_path = "{hps["original_path"]}"'),
        ('dataset_name = "cic_ids_2017_5_percent"', f'dataset_name = "{hps["dataset_name"]}"'),
        ('early_stopping_patience = max_epochs = 5', f'early_stopping_patience = max_epochs = {hps["max_epochs"]}'),
        ('USE_MULTI_SEED_EVAL = False', f'USE_MULTI_SEED_EVAL = {hps["multi_seed"]}'),
        ('loss_name = "vanilla_ce"', f'loss_name = "{hps["loss_name"]}"'),
        ('focal_gamma = 2.0', f'focal_gamma = {hps["focal_gamma"]}'),
        ('focal_alpha = weighted_class_counts', f'focal_alpha = {hps["focal_alpha"]}'),
        ('class_counts_scheme = "effective"', f'class_counts_scheme = "{hps["class_counts_scheme"]}"'),
        ('class_counts_beta = 0.999', f'class_counts_beta = {hps["class_counts_beta"]}'),
        ('class_counts_normalize = "max1"', f'class_counts_normalize = "{hps["class_counts_normalize"]}"'),
        ('cb_beta = 0.999', f'cb_beta = {hps["cb_beta"]}'),
        ('ldam_C_margin = 0.5', f'ldam_C_margin = {hps["ldam_C_margin"]}'),
        ('drw_start = 10', f'drw_start = {hps["drw_start"]}'),
        ('cb_beta_drw = 0.999', f'cb_beta_drw = {hps["cb_beta_drw"]}'),
        ('logit_adj_tau = 1.0', f'logit_adj_tau = {hps["logit_adj_tau"]}'),
    ]
    
    for old, new in replacements:
        src = src.replace(old, new)
    
            # Add comprehensive Unicode handling at the beginning
        unicode_patch = [
            "# === COMPREHENSIVE UNICODE HANDLING ===",
            "import sys",
            "import os",
            "import locale",
            "",
            "# Force UTF-8 encoding for Windows",
            "if sys.platform == \"win32\":",
            "    os.environ[\"PYTHONIOENCODING\"] = \"utf-8\"",
            "    os.environ[\"PYTHONUTF8\"] = \"1\"",
            "    # Override locale encoding",
            "    if hasattr(locale, 'setlocale'):",
            "        try:",
            "            locale.setlocale(locale.LC_ALL, 'C.UTF-8')",
            "        except:",
            "            try:",
            "                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')",
            "            except:",
            "                pass",
            "",
            "# === END UNICODE PATCH ===",
            ""
        ]
    
    # Insert the Unicode patch after the first import statement
    if 'import ' in src:
        first_import = src.find('import ')
        first_newline = src.find('\n', first_import)
        if first_newline != -1:
            patch_text = '\n'.join(unicode_patch)
            src = src[:first_newline] + patch_text + src[first_newline:]
    
    # Force CPU usage to avoid MPS/GPU issues with DGL
    # Add accelerator setting to trainer_kwargs
    if 'trainer_kwargs = {}' in src:
        src = src.replace('trainer_kwargs = {}', 'trainer_kwargs = {"accelerator": "cpu"}')
    elif 'trainer_kwargs = {"accelerator": "cpu"}' not in src:
        # If trainer_kwargs already has content, add accelerator
        src = src.replace('trainer_kwargs = {', 'trainer_kwargs = {"accelerator": "cpu", ')
    
    # Dynamic models_dict replacement - find the existing empty one and replace it
    if 'models_dict = {}' in src:
        # Find the start and end of the empty models_dict
        start = src.find('models_dict = {}')
        if start != -1:
            end = start + len('models_dict = {}')
            
            # Determine which models to include based on target_model
            if target_model == "e_gat":
                models_to_include = ["e_gat"]
            elif target_model == "e_graphsage":
                models_to_include = ["e_graphsage"]
            elif target_model == "both":
                models_to_include = ["e_gat", "e_graphsage"]
            else:
                # Handle custom model lists (future enhancement)
                models_to_include = [target_model]
            
            # Generate the new models_dict dynamically
            new_models = generate_model_dict(models_to_include, hps)
            
            src = src[:start] + new_models + src[end:]
            print(f"Replaced models_dict with: {new_models}")
        else:
            print("Warning: Could not find 'models_dict = {}' in source")
    else:
        print("Warning: No empty models_dict found to replace")
    
    # Remove the bottom execution block
    if 'if USE_MULTI_SEED_EVAL:' in src:
        src = src[:src.find('if USE_MULTI_SEED_EVAL:')]
    
    # Add the export code
    export = f"""
# === TUNER EXPORT ===
import json as _json, os as _os

try:
    print("Starting training...")
    _res = run_training_for_seed(CONFIG.seed)
    print(f"Training completed, results: {{_res}}")
    
    _out = {{
        "target_model": "{target_model}",
        "hyperparams": {repr(hps)},
        "results": _res
    }}
    
    _os.makedirs("tuning_results", exist_ok=True)
    with open(_os.path.join("tuning_results", "last_results.json"), "w", encoding="utf-8") as _f:
        _json.dump(_out, _f, ensure_ascii=False, indent=2)
    
    print("TUNER_RESULTS_JSON:", _json.dumps(_out, ensure_ascii=False))
    print("Results saved successfully")
    
except Exception as e:
    print(f"Error during training: {{e}}")
    import traceback
    traceback.print_exc()
    
    # Save error information
    _error_out = {{
        "target_model": "{target_model}",
        "hyperparams": {repr(hps)},
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    
    _os.makedirs("tuning_results", exist_ok=True)
    with open(_os.path.join("tuning_results", "last_results.json"), "w", encoding="utf-8") as _f:
        _json.dump(_error_out, _f, ensure_ascii=False, indent=2)
    
    print("Error information saved")
"""
    src = src + export
    
    patched = tmp_dir / "patched_adap_gnn_v3.py"
    try:
        patched.write_text(src, encoding="utf-8", errors='replace')
        return patched
    except Exception as e:
        print(f"Warning: Failed to write patched file: {e}")
        # Try to write with different encoding
        try:
            patched.write_text(src, encoding="latin-1", errors='replace')
            return patched
        except Exception as e2:
            print(f"Warning: Failed to write patched file with latin-1 encoding: {e2}")
            raise RuntimeError(f"Could not write patched file: {e}")

def run_one_trial(orig_py: Path, model: str, use_wandb: bool, trial=None):
    print(f"Starting trial for model: {model}")
    try:
        hps = sample_hparams(trial=trial)
        print(f"Generated hyperparameters: {hps}")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            patched = patch_python(orig_py, td_path, hps, model, use_wandb)
            print(f"Patched file created at: {patched}")
            print(f"Patched file exists: {patched.exists()}")
            print(f"Patched file size: {patched.stat().st_size if patched.exists() else 'N/A'} bytes")
            
            # Use the current Python executable (from venv if activated)
            python_exe = sys.executable
            
            # Run the patched script with comprehensive encoding environment
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
            
            # Force UTF-8 for Windows
            if os.name == 'nt':
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"
                env["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
                # Additional Windows-specific settings
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"
                # Set console code page to UTF-8
                try:
                    os.system("chcp 65001 > nul 2>&1")
                except:
                    pass
                
                # Additional environment variables for Windows
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"
                env["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"
            
            print(f"Running trial with Python: {python_exe}")
            print(f"Environment encoding vars: {dict((k, v) for k, v in env.items() if 'PYTHON' in k)}")
            
            try:
                print(f"Executing: {python_exe} {patched}")
                print(f"Working directory: {td_path}")
                print(f"Environment: {dict((k, v) for k, v in env.items() if 'PYTHON' in k)}")
                
                proc = subprocess.run([python_exe, str(patched)], cwd=str(td_path), capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
                
                # Print stderr for debugging if needed
                if proc.returncode != 0:
                    print(f"Warning: Trial returned code {proc.returncode}")
                    print(f"STDOUT: {proc.stdout}")
                    print(f"STDERR: {proc.stderr}")
                    # Don't raise error, try to continue
                else:
                    print("Subprocess executed successfully")
                    print(f"STDOUT: {proc.stdout[:200]}...")
            except Exception as e:
                print(f"Warning: Subprocess execution failed: {e}")
                # Try to continue with next trial
                return 0.0, {"results": {}, "hyperparams": hps}
            
            # If subprocess failed completely, return default values
            if 'proc' not in locals() or proc.returncode != 0:
                print("Warning: Subprocess failed, using default values")
                return 0.0, {"results": {}, "hyperparams": hps}
            
            # Additional check for encoding issues in output
            if proc.stdout and any(ord(c) > 127 for c in proc.stdout):
                print("Warning: Non-ASCII characters detected in stdout")
            if proc.stderr and any(ord(c) > 127 for c in proc.stderr):
                print("Warning: Non-ASCII characters detected in stderr")
            
            # Clean up data directory if it exists
            data_dir = td_path / "data"
            if data_dir.exists():
                import shutil
                shutil.rmtree(data_dir)
                print(f"[Trial cleanup] Removed data directory: {data_dir}")
            
            # Read result JSON
            res_path = td_path / "tuning_results" / "last_results.json"
            try:
                if res_path.exists():
                    # Try multiple encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            data = json.loads(res_path.read_text(encoding=encoding, errors='replace'))
                            print(f"Successfully read JSON with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error with {encoding} encoding: {e}")
                            continue
                    else:
                        print("Warning: Failed to read JSON with any encoding")
                        data = {"results": {}, "hyperparams": hps}
                else:
                    data = None
            except Exception as e:
                print(f"Warning: Failed to read results JSON: {e}")
                data = {"results": {}, "hyperparams": hps}
            
            # Score extraction
            if data and "results" in data:
                scores = data["results"]
                print(f"Found results: {list(scores.keys())}")
                
                # Check if results contain error information
                if "error" in data:
                    print(f"Training failed with error: {data['error']}")
                    if "traceback" in data:
                        print(f"Traceback: {data['traceback']}")
                    return 0.0, data
            else:
                print("Warning: No valid results found, using default score")
                return 0.0, {"results": {}, "hyperparams": hps}
            
            try:
                if model == "e_gat":
                    key = [k for k in scores.keys() if "e_gat" in k]
                    if key:
                        score = float(scores[key[0]])
                        print(f"Extracted e_gat score: {score}")
                    else:
                        print("Warning: No e_gat results found, using default score")
                        return 0.0, {"results": {}, "hyperparams": hps}
                elif model == "e_graphsage":
                    key = [k for k in scores.keys() if "e_graphsage" in k]
                    if key:
                        score = float(scores[key[0]])
                        print(f"Extracted e_graphsage score: {score}")
                    else:
                        print("Warning: No e_graphsage results found, using default score")
                        return 0.0, {"results": {}, "hyperparams": hps}
                else:  # both models
                    egat_key = [k for k in scores.keys() if "e_gat" in k]
                    gs_key = [k for k in scores.keys() if "e_graphsage" in k]
                    if egat_key and gs_key:
                        score = float((scores[egat_key[0]] + scores[gs_key[0]]) / 2.0)
                        print(f"Extracted combined score: {score} (e_gat: {scores[egat_key[0]]}, e_graphsage: {scores[gs_key[0]]})")
                    else:
                        print("Warning: Missing results for both models, using default score")
                        return 0.0, {"results": {}, "hyperparams": hps}
                return score, data
            except (KeyError, ValueError, TypeError) as e:
                print(f"Warning: Error extracting score: {e}")
                return 0.0, {"results": {}, "hyperparams": hps}
    except Exception as e:
        print(f"Warning: Trial failed with error: {e}")
        return 0.0, {"results": {}, "hyperparams": hps}
    
    # Safety check - ensure we always return a tuple
    print("Warning: Function reached end without explicit return, using default values")
    return 0.0, {"results": {}, "hyperparams": hps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=str, default="adap_gnn_v3.py")
    ap.add_argument("--model", type=str, default="both", choices=get_available_models())
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--optuna", action="store_true")
    ap.add_argument("--study-name", type=str, default="adap-gnn-py-search")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--python-path", type=str, default=None, help="Path to Python executable")
    ap.add_argument("--list-models", action="store_true", help="List available models and exit")
    ap.add_argument("--test-encoding", action="store_true", help="Test encoding configuration and exit")
    ap.add_argument("--test-patch", action="store_true", help="Test patching without running training")
    args = ap.parse_args()

    # Test encoding setup if requested
    if args.test_encoding:
        test_encoding_setup()
        return
    
    # Test patching if requested
    if args.test_patch:
        print("Testing patching functionality...")
        orig_py = Path(args.py).resolve()
        if not orig_py.exists():
            raise SystemExit(f"File not found: {orig_py}")
        
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            hps = sample_hparams()
            patched = patch_python(orig_py, td_path, hps, "e_gat", False)
            print(f"Patched file created at: {patched}")
            print("Patching test completed successfully")
        return
    
    # Always test encoding on Windows
    if os.name == 'nt':
        print("Windows detected - testing encoding setup...")
        test_encoding_setup()
    
    # List available models if requested
    if args.list_models:
        print("Available GNN models:")
        for model_name, config in GNN_MODELS.items():
            print(f"  {model_name}: {config['description']}")
        print("  both: Run both e_gat and e_graphsage")
        return

    orig_py = Path(args.py).resolve()
    if not orig_py.exists():
        raise SystemExit(f"File not found: {orig_py}")

    out_fp = open(args.out, "a", encoding="utf-8", errors='replace') if args.out else None

    use_optuna = args.optuna
    study = None
    if use_optuna:
        try:
            import optuna
            study = optuna.create_study(direction="maximize", study_name=args.study_name)
        except Exception as e:
            print("Optuna unavailable, using random search.", e)
            use_optuna = False

    # Use specified Python path or auto-detect
    if args.python_path:
        python_exe = args.python_path
    else:
        venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            python_exe = str(venv_python)
        else:
            python_exe = "python"

    if use_optuna:
        import optuna
        for _ in range(args.n_trials):
            trial = study.ask()
            score, data = run_one_trial(orig_py, args.model, args.wandb, trial=trial)
            study.tell(trial, score)
            if out_fp:
                out_fp.write(json.dumps({"trial": trial.number, "model": args.model, "score": score, "hyperparams": data["hyperparams"]}) + "\n")
                out_fp.flush()
        print("Best trial:", study.best_trial.number, "score:", study.best_value)
        print("Best params:", study.best_trial.params)
    else:
        best = None
        for t in range(args.n_trials):
            score, data = run_one_trial(orig_py, args.model, args.wandb, trial=None)
            if out_fp:
                out_fp.write(json.dumps({"trial": t+1, "model": args.model, "score": score, "hyperparams": data["hyperparams"]}) + "\n")
                out_fp.flush()
            print(f"[trial {t+1}/{args.n_trials}] score={score:.4f}")
            if best is None or score > best:
                best = score
        print("Best (random) score:", best)

    if out_fp:
        out_fp.close()

if __name__ == "__main__":
    main()