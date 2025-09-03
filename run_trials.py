#!/usr/bin/env python3
"""
Trial Runner Script for GNN Analysis

This script runs multiple trials with different configurations and model selections.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import traceback

# Add the src directory to the path so we can import config
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trial_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrialConfig:
    """Configuration for a single trial"""
    name: str
    config_overrides: Dict[str, Any]
    selected_models: List[str]
    description: str = ""
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Trial name cannot be empty")
        if not self.selected_models:
            raise ValueError("At least one model must be selected")

@dataclass
class TrialResult:
    """Result of a single trial"""
    trial_name: str
    config: Dict[str, Any]
    selected_models: List[str]
    start_time: str
    end_time: str
    duration: float
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    error_message: str = ""

class TrialRunner:
    """Manages running multiple trials with different configurations"""
    
    def __init__(self, base_config: Config, trials: List[TrialConfig], 
                 results_file: str = "trial_results.json", 
                 parallel: bool = False, max_workers: int = 1):
        self.base_config = base_config
        self.trials = trials
        self.results_file = results_file
        self.parallel = parallel
        self.max_workers = max_workers
        self.results: List[TrialResult] = []
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results from file if it exists"""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.results = [TrialResult(**result) for result in data]
                logger.info(f"Loaded {len(self.results)} existing results from {self.results_file}")
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
                self.results = []
    
    def save_results(self):
        """Save current results to file"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump([asdict(result) for result in self.results], f, indent=2)
            logger.info(f"Results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def create_trial_config_file(self, trial: TrialConfig) -> str:
        """Create a temporary config file for a specific trial"""
        # Create a copy of the base config
        config_dict = asdict(self.base_config)
        
        # Apply overrides
        for key, value in trial.config_overrides.items():
            if hasattr(self.base_config, key):
                config_dict[key] = value
            else:
                logger.warning(f"Unknown config key: {key}")
        
        # Create the config class
        trial_config = Config(**config_dict)
        
        # Create temporary config file
        config_content = f"""# Auto-generated config for trial: {trial.name}
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

@dataclass
class Config:
    # Dataset configuration
    dataset_name: str = "{trial_config.dataset_name}"
    original_path: str = "{trial_config.original_path}"
    folder_path: str = "{trial_config.folder_path}"
    
    # Dataset splits
    validation_size: float = {trial_config.validation_size}
    test_size: float = {trial_config.test_size}
    
    # Graph configuration
    graph_type: str = "{trial_config.graph_type}"
    window_size: int = {trial_config.window_size}
    g_type: str = "{trial_config.g_type}"
    cn_measures = {trial_config.cn_measures}
    network_features = {trial_config.network_features}
    
    # Data preprocessing flags
    with_sort_timestamp: bool = {trial_config.with_sort_timestamp}
    with_undersample_classes: bool = {trial_config.with_undersample_classes}
    use_node_features: bool = {trial_config.use_node_features}
    use_port_in_address: bool = {trial_config.use_port_in_address}
    generated_ips: bool = {trial_config.generated_ips}
    use_centralities_nfeats: bool = {trial_config.use_centralities_nfeats}
    sort_timestamp: bool = {trial_config.sort_timestamp}
    
    # Model architecture
    ndim_out: List[int] = field(default_factory=lambda: {trial_config.ndim_out})
    num_layers: int = {trial_config.num_layers}
    number_neighbors: List[int] = field(default_factory=lambda: {trial_config.number_neighbors})
    dropout: float = {trial_config.dropout}
    residual: bool = {trial_config.residual}
    multi_class: bool = {trial_config.multi_class}
    aggregation: str = "{trial_config.aggregation}"
    activation: str = "{trial_config.activation}"
    
    # Training configuration
    using_wandb: bool = {trial_config.using_wandb}
    save_top_k: int = {trial_config.save_top_k}
    early_stopping_patience: int = {trial_config.early_stopping_patience}
    max_epochs: int = {trial_config.max_epochs}
    learning_rate: float = {trial_config.learning_rate}
    weight_decay: float = {trial_config.weight_decay}
    batch_size: int = {trial_config.batch_size}
    
    # Loss function configuration
    loss_name: str = "{trial_config.loss_name}"
    
    # Focal Loss hyperparameters
    focal_alpha: Union[float, str, None] = {repr(trial_config.focal_alpha)}
    focal_gamma: float = {trial_config.focal_gamma}
    
    # Class-balanced loss hyperparameters
    class_counts_scheme: str = "{trial_config.class_counts_scheme}"
    class_counts_beta: float = {trial_config.class_counts_beta}
    class_counts_normalize: str = "{trial_config.class_counts_normalize}"
    cb_beta: float = {trial_config.cb_beta}
    
    # LDAM + DRW hyperparameters
    ldam_C_margin: float = {trial_config.ldam_C_margin}
    drw_start: int = {trial_config.drw_start}
    cb_beta_drw: float = {trial_config.cb_beta_drw}
    
    # Logit-Adjusted CE hyperparameters
    logit_adj_tau: float = {trial_config.logit_adj_tau}
    
    # Reproducibility
    seed: int = {trial_config.seed}
    
    # Advanced features flags
    use_enhanced_logging: bool = {trial_config.use_enhanced_logging}
    use_deterministic: bool = {trial_config.use_deterministic}
    use_mixed_precision: bool = {trial_config.use_mixed_precision}
    use_gradient_accumulation: bool = {trial_config.use_gradient_accumulation}
    use_extra_metrics: bool = {trial_config.use_extra_metrics}
    use_multi_seed_eval: bool = {trial_config.use_multi_seed_eval}
    use_complexity_logging: bool = {trial_config.use_complexity_logging}
    analyse_graph_metrics: bool = {trial_config.analyse_graph_metrics}
    
    def __post_init__(self):
        if self.graph_type == "flow":
            self.g_type = "flow"
        elif self.graph_type == "window":
            self.g_type = f"window_graph_{{self.window_size}}"
        
        if self.multi_class:
            self.g_type += "__multi_class"
        
        if self.use_centralities_nfeats:
            self.g_type += "__n_feats"
        
        if self.sort_timestamp:
            self.g_type += "__sorted"
        else:
            self.g_type += "__unsorted"
    
    def get_config_summary(self) -> dict:
        return {{
            'dataset_name': self.dataset_name,
            'max_epochs': self.max_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'ndim_out': self.ndim_out,
            'num_layers': self.num_layers,
            'number_neighbors': self.number_neighbors,
            'dropout': self.dropout,
            'residual': self.residual,
            'multi_class': self.multi_class,
            'aggregation': self.aggregation,
            'activation': self.activation,
            'loss_name': self.loss_name,
            'class_counts_scheme': self.class_counts_scheme,
            'class_counts_beta': self.class_counts_beta,
            'class_counts_normalize': self.class_counts_normalize,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'cb_beta': self.cb_beta,
            'ldam_C_margin': self.ldam_C_margin,
            'drw_start': self.drw_start,
            'cb_beta_drw': self.cb_beta_drw,
            'logit_adj_tau': self.logit_adj_tau,
            'validation_size': self.validation_size,
            'test_size': self.test_size,
            'graph_type': self.graph_type,
            'g_type': self.g_type,
            'cn_measures': self.cn_measures,
            'network_features': self.network_features,
            'use_enhanced_logging': self.use_enhanced_logging,
            'use_deterministic': self.use_deterministic,
            'use_mixed_precision': self.use_mixed_precision,
            'use_gradient_accumulation': self.use_gradient_accumulation,
            'use_extra_metrics': self.use_extra_metrics,
            'use_multi_seed_eval': self.use_multi_seed_eval,
            'use_complexity_logging': self.use_complexity_logging,
            'analyse_graph_metrics': self.analyse_graph_metrics,
        }}
    
    def print_config_summary(self):
        print('Configuration Summary:')
        config_summary = self.get_config_summary()
        for key, value in config_summary.items():
            print(f"  {{key}}: {{value}}")

# Default configuration instance
CONFIG = Config()
"""
        
        # Create temporary file
        temp_config_file = f"temp_config_{trial.name.replace(' ', '_').replace('-', '_')}.py"
        with open(temp_config_file, 'w') as f:
            f.write(config_content)
        
        return temp_config_file
    
    def create_trial_main_file(self, trial: TrialConfig, config_file: str) -> str:
        """Create a modified main.py file for the trial with specific model selection"""
        # Read the original main.py
        with open('main.py', 'r') as f:
            main_content = f.read()
        
        # Create the model selection code
        model_selection_code = f"""
    # Trial-specific model selection for: {trial.name}
    selected_models = {trial.selected_models}
    
    # Filter models based on trial selection
    my_models = {{k: all_models[k] for k in selected_models if k in all_models}}
    
    if not my_models:
        raise ValueError(f"No valid models found in selected_models: {{selected_models}}. Available models: {{list(all_models.keys())}})
"""
        
        # Replace the model selection section in main.py
        # Find the line with "selected_models = [f"e_graphsage_{config.aggregation}"]"
        old_selection = '    selected_models = [f"e_graphsage_{config.aggregation}"]'
        new_selection = f'    # Original selection replaced by trial configuration\n    # {old_selection}'
        
        # Find the line with "my_models = {k: all_models[k] for k in selected_models}"
        old_models = '    my_models = {k: all_models[k] for k in selected_models}'
        
        # Replace both sections
        main_content = main_content.replace(old_selection, new_selection)
        main_content = main_content.replace(old_models, model_selection_code)
        
        # Create temporary main file
        temp_main_file = f"temp_main_{trial.name.replace(' ', '_').replace('-', '_')}.py"
        with open(temp_main_file, 'w') as f:
            f.write(main_content)
        
        return temp_main_file
    
    def run_single_trial(self, trial: TrialConfig) -> TrialResult:
        """Run a single trial and return the result"""
        start_time = datetime.now()
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Starting trial: {trial.name}")
        logger.info(f"Config overrides: {trial.config_overrides}")
        logger.info(f"Selected models: {trial.selected_models}")
        
        # Create temporary files
        config_file = self.create_trial_config_file(trial)
        main_file = self.create_trial_main_file(trial, config_file)
        
        try:
            # Set environment variable to use our temporary config
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
            
            # Run the trial
            cmd = [sys.executable, main_file]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.getcwd(),
                timeout=7200  # 2 hour timeout
            )
            
            end_time = datetime.now()
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = (end_time - start_time).total_seconds()
            
            # Create trial result
            trial_result = TrialResult(
                trial_name=trial.name,
                config=trial.config_overrides,
                selected_models=trial.selected_models,
                start_time=start_time_str,
                end_time=end_time_str,
                duration=duration,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=result.returncode == 0,
                error_message=result.stderr if result.returncode != 0 else ""
            )
            
            if trial_result.success:
                logger.info(f"Trial {trial.name} completed successfully in {duration:.2f} seconds")
            else:
                logger.error(f"Trial {trial.name} failed with exit code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
            
            return trial_result
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Trial {trial.name} timed out after {duration:.2f} seconds")
            
            return TrialResult(
                trial_name=trial.name,
                config=trial.config_overrides,
                selected_models=trial.selected_models,
                start_time=start_time_str,
                end_time=end_time_str,
                duration=duration,
                exit_code=-1,
                stdout="",
                stderr="Trial timed out",
                success=False,
                error_message="Trial timed out after 2 hours"
            )
            
        except Exception as e:
            end_time = datetime.now()
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Trial {trial.name} failed with exception: {e}")
            logger.error(traceback.format_exc())
            
            return TrialResult(
                trial_name=trial.name,
                config=trial.config_overrides,
                selected_models=trial.selected_models,
                start_time=start_time_str,
                end_time=end_time_str,
                duration=duration,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                success=False,
                error_message=str(e)
            )
            
        finally:
            # Clean up temporary files
            try:
                os.remove(config_file)
                os.remove(main_file)
            except Exception as e:
                logger.warning(f"Could not clean up temporary files: {e}")
    
    def run_trials(self):
        """Run all trials"""
        logger.info(f"Starting {len(self.trials)} trials")
        
        if self.parallel and self.max_workers > 1:
            self._run_trials_parallel()
        else:
            self._run_trials_sequential()
        
        # Save final results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def _run_trials_sequential(self):
        """Run trials sequentially"""
        for i, trial in enumerate(self.trials, 1):
            logger.info(f"Running trial {i}/{len(self.trials)}: {trial.name}")
            
            # Check if trial was already completed
            if any(r.trial_name == trial.name and r.success for r in self.results):
                logger.info(f"Trial {trial.name} already completed successfully, skipping")
                continue
            
            # Run the trial
            result = self.run_single_trial(trial)
            self.results.append(result)
            
            # Save results after each trial
            self.save_results()
            
            # Small delay between trials
            time.sleep(2)
    
    def _run_trials_parallel(self):
        """Run trials in parallel using multiprocessing"""
        logger.info(f"Running trials in parallel with {self.max_workers} workers")
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            # Submit all trials
            futures = []
            for trial in self.trials:
                # Check if trial was already completed
                if any(r.trial_name == trial.name and r.success for r in self.results):
                    logger.info(f"Trial {trial.name} already completed successfully, skipping")
                    continue
                
                future = pool.apply_async(self.run_single_trial, (trial,))
                futures.append((trial, future))
            
            # Collect results as they complete
            for i, (trial, future) in enumerate(futures, 1):
                try:
                    result = future.get(timeout=7200)  # 2 hour timeout per trial
                    self.results.append(result)
                    logger.info(f"Completed trial {i}/{len(futures)}: {trial.name}")
                    
                    # Save results after each completed trial
                    self.save_results()
                    
                except Exception as e:
                    logger.error(f"Trial {trial.name} failed: {e}")
                    # Create a failed result
                    failed_result = TrialResult(
                        trial_name=trial.name,
                        config=trial.config_overrides,
                        selected_models=trial.selected_models,
                        start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        duration=0,
                        exit_code=-1,
                        stdout="",
                        stderr=str(e),
                        success=False,
                        error_message=str(e)
                    )
                    self.results.append(failed_result)
                    self.save_results()
    
    def print_summary(self):
        """Print a summary of all trial results"""
        print("\n" + "="*80)
        print("TRIAL RUNNER SUMMARY")
        print("="*80)
        
        total_trials = len(self.trials)
        successful_trials = sum(1 for r in self.results if r.success)
        failed_trials = total_trials - successful_trials
        
        print(f"Total trials: {total_trials}")
        print(f"Successful: {successful_trials}")
        print(f"Failed: {failed_trials}")
        print(f"Success rate: {successful_trials/total_trials*100:.1f}%")
        
        if self.results:
            total_duration = sum(r.duration for r in self.results if r.success)
            avg_duration = total_duration / successful_trials if successful_trials > 0 else 0
            print(f"Total runtime: {total_duration/3600:.2f} hours")
            print(f"Average trial duration: {avg_duration/60:.2f} minutes")
        
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in self.results:
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            duration_str = f"{result.duration/60:.1f} min" if result.duration > 0 else "N/A"
            print(f"{status:<12} {result.trial_name:<30} {duration_str:>10}")
            
            if not result.success and result.error_message:
                print(f"  Error: {result.error_message[:100]}...")
        
        print("="*80)

def load_trial_configs(config_file: str) -> List[TrialConfig]:
    """Load trial configurations from a Python file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Trial config file not found: {config_file}")
    
    # Import the config file
    import importlib.util
    spec = importlib.util.spec_from_file_location("trial_configs", config_file)
    trial_configs_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial_configs_module)
    
    # Get the trials list
    if not hasattr(trial_configs_module, 'trials'):
        raise ValueError("Trial config file must contain a 'trials' list")
    
    return trial_configs_module.trials

def main():
    parser = argparse.ArgumentParser(description="Run multiple GNN analysis trials")
    parser.add_argument('--config-file', required=True, 
                       help='Python file containing trial configurations')
    parser.add_argument('--results-file', default='trial_results.json',
                       help='File to save trial results (default: trial_results.json)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run trials in parallel')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum number of parallel workers (default: 1)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous results file')
    
    args = parser.parse_args()
    
    try:
        # Load trial configurations
        logger.info(f"Loading trial configurations from {args.config_file}")
        trials = load_trial_configs(args.config_file)
        logger.info(f"Loaded {len(trials)} trial configurations")
        
        # Load base configuration
        from config import CONFIG
        base_config = CONFIG
        
        # Create trial runner
        runner = TrialRunner(
            base_config=base_config,
            trials=trials,
            results_file=args.results_file,
            parallel=args.parallel,
            max_workers=args.max_workers
        )
        
        # Run trials
        runner.run_trials()
        
        logger.info("All trials completed!")
        
    except Exception as e:
        logger.error(f"Trial runner failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
