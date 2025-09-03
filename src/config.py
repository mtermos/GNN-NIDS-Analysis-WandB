from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import os

@dataclass
class Config:
    # =========================
    # Dataset configuration
    dataset_name: str = "cic_ids_2017_5_percent"
    original_path: str = "testing_dfs/cic_ids_2017_5_percent.parquet"
    folder_path: str = "data"

    # Dataset splits
    validation_size: float = 0.1
    test_size: float = 0.1

    # Graph configuration
    graph_type: str = "flow"  # "flow", "window", "line"
    window_size: int = 500
    g_type: str = ""
    cn_measures = ["betweenness", "degree", "pagerank", "closeness", "k_truss"]
    network_features = ['src_betweenness', 'dst_betweenness', 'src_degree', 'dst_degree', 'src_pagerank', 'dst_pagerank', 'src_closeness', 'dst_closeness', 'src_k_truss', 'dst_k_truss']

    # Data preprocessing flags
    with_sort_timestamp: bool = False
    with_undersample_classes: bool = False
    use_node_features: bool = False
    use_port_in_address: bool = False
    generated_ips: bool = False
    use_centralities_nfeats: bool = False
    sort_timestamp: bool = False

    # Model architecture
    selected_models: List[str] = field(default_factory=lambda: ["e_graphsage"])
    ndim_out: List[int] = field(default_factory=lambda: [128, 128])
    num_layers: int = 2
    number_neighbors: List[int] = field(default_factory=lambda: [25, 10])
    dropout: float = 0.5
    residual: bool = True
    edge_update: bool = True
    multi_class: bool = True
    aggregation: str = "mean"
    activation: str = "relu"

    # Training configuration
    using_wandb: bool = False
    save_top_k: int = 5
    early_stopping_patience: int = 5
    max_epochs: int = 5
    learning_rate: float = 0.005
    weight_decay: float = 0.01
    batch_size: int = 1

    # Loss function configuration
    loss_name: str = "vanilla_ce"  # ["vanilla_ce","ce_cb","focal","ldam_drw","logit_adj","balanced_softmax"]

    # Focal Loss hyperparameters
    focal_alpha: Union[float, str, None] = "weighted_class_counts"  # None | scalar | "weighted_class_counts"
    focal_gamma: float = 2.0

    # Class-balanced loss hyperparameters
    class_counts_scheme: str = "effective"  # "effective" | "inverse" | "median" | "sqrt_inv"
    class_counts_beta: float = 0.999
    class_counts_normalize: str = "max1"
    cb_beta: float = 0.999

    # LDAM + DRW hyperparameters
    ldam_C_margin: float = 0.5
    drw_start: int = 10  # epoch to turn on class-balanced weights
    cb_beta_drw: float = 0.999  # beta for DRW phase

    # Logit-Adjusted CE hyperparameters
    logit_adj_tau: float = 1.0

    # Reproducibility
    seed: int = 42

    # Advanced features flags (all default to False unless otherwise specified)
    use_enhanced_logging: bool = False           # Enable enhanced logging
    use_deterministic: bool = False              # Enable deterministic training for reproducibility
    use_mixed_precision: bool = False            # Enable mixed precision training
    use_gradient_accumulation: bool = False      # Enable gradient accumulation
    use_extra_metrics: bool = False              # Compute additional metrics: accuracy, precision, recall
    use_multi_seed_eval: bool = False            # Run multiple seeds and report mean±std
    use_complexity_logging: bool = False         # Log parameter counts and training times
    analyse_graph_metrics: bool = True           # Compute attacker proportion and degree distribution entropy
    
    def __post_init__(self):
        """Set derived configuration values after initialization"""
        # Set g_type based on other parameters
        if self.graph_type == "flow":
            self.g_type = "flow"
        elif self.graph_type == "window":
            self.g_type = f"window_graph_{self.window_size}"
        
        if self.multi_class:
            self.g_type += "__multi_class"
        
        if self.use_centralities_nfeats:
            self.g_type += "__n_feats"
        
        if self.sort_timestamp:
            self.g_type += "__sorted"
        else:
            self.g_type += "__unsorted"
    
    def get_config_summary(self) -> dict:
        """Return a summary of the configuration for logging"""
        return {
            'dataset_name': self.dataset_name,
            'max_epochs': self.max_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'selected_models': self.selected_models,
            'ndim_out': self.ndim_out,
            'num_layers': self.num_layers,
            'number_neighbors': self.number_neighbors,
            'dropout': self.dropout,
            'residual': self.residual,
            'edge_update': self.edge_update,
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
        }
    
    def print_config_summary(self):
        """Print configuration summary"""
        print('Configuration Summary:')
        config_summary = self.get_config_summary()
        for key, value in config_summary.items():
            print(f"  {key}: {value}")

# Default configuration instance
CONFIG = Config()
