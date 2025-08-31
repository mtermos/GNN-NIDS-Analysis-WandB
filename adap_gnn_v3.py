# %% [markdown]
# # Notebook Guide
# 
# This notebook extends the baseline GNN intrusion detection pipeline to address reviewer feedback. New capabilities (guarded by flags) include additional GNN variants, multi‑seed evaluation for statistical significance, extra metrics, computational complexity logging and graph metric analysis. Defaults preserve the original behaviour.
# 
# **Purpose**: Train and evaluate adaptive GNN models on the CIC‑IDS‑2017 dataset while providing optional analyses requested by reviewers.
# 
# **High‑level pipeline** (unchanged):
# 1. Environment Setup
# 2. Configuration
# 3. Imports & Utilities
# 4. Data Loading & Preprocessing
# 5. Graph Preparation
# 6. Model Definition
# 7. Training & Evaluation
# 8. Artifacts & Logging
# 
# Optional extensions are controlled via flags defined in the Advanced Configuration section.
# 

# %% [markdown]
# ## Configuration
# 
# Define hyperparameters and paths.
# 
# **Focal Loss**: When `use_focal_loss=True`, the model uses Focal Loss instead of CrossEntropyLoss to address class imbalance. 
# - `focal_alpha`: Weighting factor for rare class (default: 0.25)
# - `focal_gamma`: Focusing parameter that down-weights easy examples (default: 2.0)
# 
# Focal Loss is particularly useful for intrusion detection datasets where attack classes are often underrepresented.

# %%
dataset_name = "cic_ids_2017_5_percent"
original_path = "testing_dfs\cic_ids_2017_5_percent.parquet"
using_wandb = True
save_top_k = 5
early_stopping_patience = max_epochs = 5
# early_stopping_patience = 200 all - epochs = 500 - lr = 0.005
learning_rate = 0.005
weight_decay = 0.01
# ndim_out = [32, 32]
ndim_out = [128, 128]
num_layers = 2
number_neighbors = [25, 10]
dropout = 0.5
residual = True
multi_class = True
aggregation = "mean"

loss_name = "vanilla_ce"          # ["vanilla_ce","ce_cb","focal","ldam_drw","logit_adj","balanced_softmax"]

# Focal
focal_gamma = 2.0
weighted_class_counts = "weighted_class_counts" 
focal_alpha = weighted_class_counts       # None | scalar | "weighted_class_counts"

class_counts_scheme = "effective"       # "effective" | "inverse" | "median" | "sqrt_inv"
class_counts_beta = 0.999
class_counts_normalize = "max1"

# Class-Balanced (Effective Number)
cb_beta = 0.999

# LDAM + DRW
ldam_C_margin = 0.5
drw_start = 10                # epoch to turn on class-balanced weights
cb_beta_drw = 0.999             # beta for DRW phase

# Logit-Adjusted CE (priors)
logit_adj_tau = 1.0

with_sort_timestamp = False
with_undersample_classes = False

use_node_features = False

use_port_in_address = False

generated_ips = False

use_centralities_nfeats = False

sort_timestamp = True

validation_size = 0.1
test_size = 0.1

folder_path = "data"

graph_type = "flow"
# graph_type = "window"
# graph_type = "line"

window_size= 500

g_type = ""
if graph_type == "flow":
    g_type = "flow"
elif graph_type == "window":
    g_type = f"window_graph_{window_size}"

if multi_class:
    g_type += "__multi_class"

if use_centralities_nfeats:
    g_type += "__n_feats"

if sort_timestamp:
    g_type += "__sorted"
else:
    g_type += "__unsorted"

# Seed for reproducibility
seed = 42


# %%
print('Configuration Summary:')
config_summary = {
    'dataset_name': dataset_name,
    'max_epochs': max_epochs,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'ndim_out': ndim_out,
    'num_layers': num_layers,
    'number_neighbors': number_neighbors,
    'dropout': dropout,
    'residual': residual,
    'multi_class': multi_class,
    'aggregation': aggregation,
    'loss_name': loss_name,
    'class_counts_scheme': class_counts_scheme,
    'class_counts_beta': class_counts_beta,
    'class_counts_normalize': class_counts_normalize,
    'focal_alpha': focal_alpha,
    'focal_gamma': focal_gamma,
    'cb_beta': cb_beta,
    'ldam_C_margin': ldam_C_margin,
    'drw_start': drw_start,
    'cb_beta_drw': cb_beta_drw,
    'logit_adj_tau': logit_adj_tau,
    'validation_size': validation_size,
    'test_size': test_size,
    'graph_type': graph_type,
}
print(config_summary)

# %% [markdown]
# ### Advanced Configuration
# 
# Encapsulate all hyperparameters and expose optional reviewer‑requested features via flags. All flags default to `False`.

# %%
from dataclasses import dataclass, field

@dataclass
class Config:
    dataset_name: str = dataset_name
    original_path: str = original_path
    using_wandb: bool = using_wandb
    save_top_k: int = save_top_k
    early_stopping_patience: int = early_stopping_patience
    max_epochs: int = max_epochs
    learning_rate: float = learning_rate
    weight_decay: float = weight_decay
    ndim_out: list = field(default_factory=lambda: ndim_out.copy())
    num_layers: int = num_layers
    number_neighbors: list = field(default_factory=lambda: number_neighbors.copy())
    dropout: float = dropout
    residual: bool = residual
    multi_class: bool = multi_class
    aggregation: str = aggregation
    # Focal Loss hyperparameters
    loss_name: str = loss_name
    class_counts_scheme: str = class_counts_scheme
    class_counts_beta: float = class_counts_beta
    class_counts_normalize: str = class_counts_normalize
    focal_alpha: float = focal_alpha
    focal_gamma: float = focal_gamma
    cb_beta: float = cb_beta
    ldam_C_margin: float = ldam_C_margin
    drw_start: int = drw_start
    cb_beta_drw: float = cb_beta_drw
    logit_adj_tau: float = logit_adj_tau
    with_sort_timestamp: bool = with_sort_timestamp
    with_undersample_classes: bool = with_undersample_classes
    use_node_features: bool = use_node_features
    use_port_in_address: bool = use_port_in_address
    generated_ips: bool = generated_ips
    use_centralities_nfeats: bool = use_centralities_nfeats
    sort_timestamp: bool = sort_timestamp
    validation_size: float = validation_size
    test_size: float = test_size
    folder_path: str = folder_path
    graph_type: str = graph_type
    window_size: int = window_size
    g_type: str = g_type
    seed: int = seed

CONFIG = Config()

# Reviewer‑requested flags (default off)
USE_ENHANCED_LOGGING = False  # reuse from v2
USE_DETERMINISTIC = False     # reuse from v2
USE_MIXED_PRECISION = False   # reuse from v2
USE_GRADIENT_ACCUMULATION = False  # reuse from v2
# New flags for reviewer suggestions
USE_EXTRA_METRICS = False         # compute accuracy, precision, recall
USE_MULTI_SEED_EVAL = False       # run multiple seeds and report mean±std
USE_COMPLEXITY_LOGGING = False    # log parameter counts and training times
ANALYSE_GRAPH_METRICS = True     # compute proportion of attackers and entropy of degree distribution



# Import libraries
# %%
import itertools
import json
import math
import os
import warnings
import pickle
import random
import socket
import struct
import time
import timeit
from collections import defaultdict
from functools import wraps

import dgl
import dgl.function as fn
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from dgl import from_networkx
from dgl.nn.pytorch import EdgeWeightNorm
from scipy.stats import entropy, skew
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import class_weight



os.environ["DGLBACKEND"] = "pytorch"

wandb.login(key="")

run_dtime = time.strftime("%Y%m%d-%H%M%S")

# Focal Loss implementation
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    Args:
        alpha (float): Weighting factor for rare class (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Reduction method ('none', 'mean', 'sum')
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (N, C) where C is the number of classes
            targets: Ground truth labels (N,) where values are in [0, C-1]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def _normalize_alpha(alpha: torch.Tensor, mode: str | None = "max1") -> torch.Tensor:
    if mode is None:
        return alpha
    if mode == "mean1":  # average weight = 1
        return alpha * (alpha.numel() / alpha.sum())
    if mode == "max1":   # strongest class = 1 (keeps focal <= CE if gamma>=0)
        return alpha / alpha.max()
    raise ValueError(mode)


def alpha_from_counts(
    counts: torch.Tensor,
    scheme: str = "effective",  # ["inverse","effective","median","sqrt_inv"]
    beta: float = 0.999,        # only used by "effective"
    normalize: str | None = "max1",
) -> torch.Tensor:
    counts = counts.to(torch.float32).clamp_min_(1)
    if scheme == "inverse":
        alpha = 1.0 / counts
    elif scheme == "effective":
        # Class-Balanced (Cui et al. CVPR'19): (1 - beta) / (1 - beta^n_c)
        eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32, device=counts.device), counts)
        alpha = (1.0 - beta) / eff
    elif scheme == "median":
        alpha = torch.median(counts) / counts
    elif scheme == "sqrt_inv":
        alpha = 1.0 / torch.sqrt(counts)
    else:
        raise ValueError(scheme)
    # first bring mean near 1 (nice for comparing schemes)
    alpha = alpha / alpha.mean()
    # then apply chosen normalization
    return _normalize_alpha(alpha, normalize)


def seed_everything_deterministic(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    try:
        torch.cuda.manual_seed_all(seed_value)
    except Exception:
        pass
    pl.seed_everything(seed_value, workers=True)

if USE_DETERMINISTIC:
    seed_everything_deterministic(CONFIG.seed)

# %% [markdown]
# ## Utility Definitions
# 
# Helper functions and classes.

# %% [markdown]
# ### Dataset Utilities

# %%
class DatasetInfo:
    def __init__(
            self,
            name,
            file_type,

            # Key Columns names
            src_ip_col,
            src_port_col,
            dst_ip_col,
            dst_port_col,
            flow_id_col,
            timestamp_col,
            label_col,
            class_col,

            class_num_col=None,
            timestamp_format=None,

            centralities_set=1,

            # Columns to be dropped from the dataset during preprocessing.
            drop_columns=[],

            # Columns to be dropped from the dataset during preprocessing.
            weak_columns=[],
    ):

        self.name = name
        self.file_type = file_type
        self.src_ip_col = src_ip_col
        self.src_port_col = src_port_col
        self.dst_ip_col = dst_ip_col
        self.dst_port_col = dst_port_col
        self.flow_id_col = flow_id_col
        self.timestamp_col = timestamp_col
        self.timestamp_format = timestamp_format
        self.label_col = label_col
        self.class_col = class_col
        self.centralities_set = centralities_set
        self.class_num_col = class_num_col
        self.drop_columns = drop_columns
        self.weak_columns = weak_columns


datasets_list = [
    DatasetInfo(name="cic_ton_iot_5_percent",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Flow IAT Mean', 'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Bwd Header Len', 'Tot Bwd Pkts', 'Bwd Pkt Len Mean', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
                              'CWE Flag Count', 'Bwd IAT Tot', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Min', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Pkt Len Var', 'FIN Flag Cnt', 'Bwd IAT Mean', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Tot', 'PSH Flag Cnt', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ton_iot",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="%d/%m/%Y %I:%M:%S %p",

                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Cnt', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Subflow Bwd Pkts', 'Active Mean', 'Active Std', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Tot', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd Seg Size Avg', 'CWE Flag Count', 'FIN Flag Cnt',
                              'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Flow Pkts/s', 'Fwd IAT Mean', 'Fwd IAT Tot', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Fwd Seg Size Avg', 'Idle Mean', 'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'Tot Bwd Pkts', 'TotLen Bwd Pkts']
                ),
    DatasetInfo(name="cic_ids_2017_5_percent",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                centralities_set=2,
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_ids_2017",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Fwd IAT Min',  'Idle Max', 'Flow IAT Mean',  'Protocol',   'Fwd Pkt Len Max', 'Flow IAT Max', 'Active Std', 'Subflow Fwd Pkts', 'Bwd Pkt Len Mean', 'Tot Bwd Pkts', 'Pkt Size Avg',
                              'Subflow Bwd Pkts', 'Bwd IAT Std', 'Fwd IAT Mean', 'Fwd Pkt Len Std', 'Pkt Len Mean', 'Flow IAT Std', 'Fwd URG Flags', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max',  'Pkt Len Var',  'Tot Fwd Pkts', 'Bwd IAT Mean', 'TotLen Fwd Pkts', 'Fwd PSH Flags', 'Idle Mean', 'Pkt Len Max', 'Flow Pkts/s', 'Flow Duration', 'Pkt Len Std', 'Fwd IAT Max',  'Fwd IAT Tot', 'RST Flag Cnt', 'Subflow Bwd Byts', 'Active Mean', 'Bwd Pkt Len Std', 'Fwd Pkt Len Mean']
                ),
    DatasetInfo(name="cic_bot_iot",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['Active Max', 'Active Mean', 'Bwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Tot', 'Bwd PSH Flags', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Min', 'Bwd Pkt Len Std', 'Bwd Pkts/b Avg', 'Bwd URG Flags', 'CWE Flag Count', 'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Std', 'Flow Pkts/s', 'Fwd Blk Rate Avg',
                              'Fwd Byts/b Avg', 'Fwd Header Len', 'Fwd IAT Max', 'Fwd IAT Mean', 'Fwd PSH Flags', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Min', 'Fwd Pkts/b Avg', 'Fwd Seg Size Min', 'Fwd URG Flags', 'Idle Max', 'Idle Mean', 'Init Fwd Win Byts', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Size Avg', 'Subflow Fwd Pkts', 'Tot Bwd Pkts', 'Tot Fwd Pkts', 'TotLen Bwd Pkts', 'TotLen Fwd Pkts']
                ),
    DatasetInfo(name="cic_ton_iot_modified",
                file_type="parquet",
                src_ip_col="Src IP",
                src_port_col="Src Port",
                dst_ip_col="Dst IP",
                dst_port_col="Dst Port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Flow ID", "Src IP", "Dst IP",
                              "Timestamp", "Src Port", "Dst Port", "Attack"],
                weak_columns=['ACK Flag Cnt', 'Active Mean', 'Active Std', 'Bwd Byts/b Avg', 'Bwd Header Len', 'Bwd IAT Mean', 'Bwd IAT Tot', 'Bwd PSH Flags', 'Bwd Pkt Len Max', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Bwd Seg Size Avg', 'Bwd URG Flags', 'CWE Flag Count', 'ECE Flag Cnt', 'FIN Flag Cnt', 'Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Flow IAT Std', 'Flow Pkts/s',
                              'Fwd Blk Rate Avg', 'Fwd Byts/b Avg', 'Fwd Header Len', 'Fwd IAT Mean', 'Fwd IAT Tot', 'Fwd Pkt Len Max', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Fwd Pkts/b Avg', 'Fwd Seg Size Avg', 'Fwd URG Flags', 'Idle Mean', 'PSH Flag Cnt', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg', 'Subflow Bwd Pkts', 'Tot Bwd Pkts', 'Tot Fwd Pkts', 'TotLen Bwd Pkts', 'URG Flag Cnt']
                ),

    DatasetInfo(name="nf_ton_iotv2_modified",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"],
                weak_columns=['CLIENT_TCP_FLAGS', 'DURATION_IN', 'ICMP_TYPE', 'IN_BYTES', 'LONGEST_FLOW_PKT',
                              'MAX_TTL', 'MIN_TTL', 'PROTOCOL', 'RETRANSMITTED_OUT_BYTES', 'TCP_FLAGS', 'TCP_WIN_MAX_IN'],
                ),

    DatasetInfo(name="ccd_inid_modified",
                file_type="parquet",
                src_ip_col="src_ip",
                src_port_col="src_port",
                dst_ip_col="dst_ip",
                dst_port_col="dst_port",
                flow_id_col="id",
                timestamp_col=None,
                label_col="traffic_type",
                class_col="atk_type",
                class_num_col="Class",
                timestamp_format=None,
                drop_columns=["id", "src_ip", "src_port",
                              "dst_ip", "dst_port", "atk_type", 'Unnamed: 0', 'src_ip_is_private', 'dst_ip_is_private',
                              'expiration_id', 'splt_direction', 'splt_ps', 'splt_piat_ms'],
                weak_columns=['application_name', 'bidirectional_ack_packets', 'bidirectional_bytes', 'bidirectional_cwr_packets', 'bidirectional_duration_ms', 'bidirectional_ece_packets', 'bidirectional_fin_packets', 'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms', 'bidirectional_max_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_mean_ps', 'bidirectional_min_piat_ms', 'bidirectional_packets', 'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_stddev_piat_ms', 'bidirectional_stddev_ps', 'bidirectional_syn_packets',
                              'bidirectional_urg_packets', 'dst2src_bytes', 'dst2src_cwr_packets', 'dst2src_duration_ms', 'dst2src_ece_packets', 'dst2src_first_seen_ms', 'dst2src_last_seen_ms', 'dst2src_max_ps', 'dst2src_mean_ps', 'dst2src_min_piat_ms', 'dst2src_packets', 'dst2src_stddev_piat_ms', 'dst2src_stddev_ps', 'dst2src_urg_packets', 'ip_version', 'src2dst_bytes', 'src2dst_cwr_packets', 'src2dst_duration_ms', 'src2dst_ece_packets', 'src2dst_first_seen_ms', 'src2dst_mean_ps', 'src2dst_min_piat_ms', 'src2dst_packets', 'src2dst_syn_packets', 'src2dst_urg_packets', 'vlan_id'],
                ),

    DatasetInfo(name="nf_uq_nids_modified",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack", "Dataset"],
                weak_columns=[],
                ),

    DatasetInfo(name="edge_iiot",
                file_type="parquet",
                src_ip_col="ip.src_host",
                src_port_col=None,
                dst_ip_col="ip.dst_host",
                dst_port_col=None,
                flow_id_col=None,
                timestamp_col="frame.time",
                label_col="Attack_label",
                class_col="Attack_type",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["ip.src_host", "ip.dst_host", "http.tls_port",
                              "dns.qry.type", "mqtt.msg_decoded_as", "frame.time", "Attack_type"],
                weak_columns=["tcp.flags", "mqtt.conflags", "mqtt.conflag.cleansess", "mbtcp.trans_id", "mqtt.hdrflags", "mqtt.msg",
                              "mqtt.len", "dns.retransmit_request", "http.request.method", "icmp.unused", "mbtcp.len", "mqtt.proto_len", "arp.opcode"]
                ),

    DatasetInfo(name="nf_cse_cic_ids2018",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"],
                weak_columns=['IN_BYTES', 'OUT_BYTES']
                ),

    DatasetInfo(name="nf_bot_iotv2",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack"],
                weak_columns=["RETRANSMITTED_IN_BYTES", "MIN_TTL", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "PROTOCOL", "SERVER_TCP_FLAGS", "CLIENT_TCP_FLAGS",
                              "LONGEST_FLOW_PKT", "NUM_PKTS_512_TO_1024_BYTES", "RETRANSMITTED_OUT_BYTES", "IN_PKTS", "TCP_FLAGS", "IN_BYTES", "ICMP_TYPE"]
                ),

    DatasetInfo(name="nf_uq_nids",
                file_type="parquet",
                src_ip_col="IPV4_SRC_ADDR",
                src_port_col="L4_SRC_PORT",
                dst_ip_col="IPV4_DST_ADDR",
                dst_port_col="L4_DST_PORT",
                flow_id_col=None,
                timestamp_col=None,
                label_col="Label",
                class_col="Attack",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["IPV4_SRC_ADDR", "L4_SRC_PORT",
                              "IPV4_DST_ADDR", "L4_DST_PORT", "Attack", "Dataset"],
                weak_columns=[],
                ),

    DatasetInfo(name="x_iiot",
                file_type="parquet",
                src_ip_col="Scr_IP",
                src_port_col="Scr_port",
                dst_ip_col="Des_IP",
                dst_port_col="Des_port",
                flow_id_col="Flow ID",
                timestamp_col="Timestamp",
                label_col="class3",
                class_col="class2",
                class_num_col="Class",
                timestamp_format="mixed",
                drop_columns=["Scr_IP", "Scr_port", "Des_IP",
                              "Des_port", "Timestamp", "Date", "class1", "class2"],
                weak_columns=["Process_activity", "Login_attempt", "is_syn_only", "Avg_iowait_time", "Avg_num_Proc/s", "Duration", "Des_pkts_ratio",
                              "is_SYN_with_RST", "Succesful_login", "OSSEC_alert", "is_pure_ack", "Conn_state", "Bad_checksum", "File_activity", "Avg_rtps", "Is_SYN_ACK"]
                ),
]


datasets = {dataset.name: dataset for dataset in datasets_list}

cn_measures = [
    ["betweenness", "local_betweenness", "degree", "local_degree",
     "eigenvector", "closeness", "pagerank", "local_pagerank", "k_core", "k_truss", "Comm"],
    ["betweenness", "global_betweenness", "degree", "global_degree",
     "eigenvector", "closeness", "pagerank", "global_pagerank", "k_core", "k_truss", "mv"],
    ["betweenness", "local_betweenness", "pagerank",
        "local_pagerank", "k_core", "k_truss", "Comm"],
    ["betweenness", "global_betweenness", "pagerank",
        "global_pagerank", "k_core", "k_truss", "mv"]
]

network_features = [
    ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_degree', 'dst_degree', 'src_local_degree', 'dst_local_degree', 'src_eigenvector',
     'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm'],
    ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_degree', 'dst_degree', 'src_global_degree', 'dst_global_degree', 'src_eigenvector',
     'dst_eigenvector', 'src_closeness', 'dst_closeness', 'src_pagerank', 'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv'],
    ['src_betweenness', 'dst_betweenness', 'src_local_betweenness', 'dst_local_betweenness', 'src_pagerank',
     'dst_pagerank', 'src_local_pagerank', 'dst_local_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_Comm', 'dst_Comm'],
    ['src_betweenness', 'dst_betweenness', 'src_global_betweenness', 'dst_global_betweenness', 'src_pagerank',
     'dst_pagerank', 'src_global_pagerank', 'dst_global_pagerank', 'src_k_core', 'dst_k_core', 'src_k_truss', 'dst_k_truss', 'src_mv', 'dst_mv'],

]


# dataset_utils.py
def clean_dataset(df, timestamp_col=None, flow_id_col=None):
    print(f"==>> original df.shape[0]: {df.shape[0]}")
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(axis=0, how='any', inplace=True)
    # df.dropna(axis=0, how='any', inplace=True, subset=list(set(
    #     df.columns) - set(id_columns)))
    print(f"==>> after drop na df.shape[0]: {df.shape[0]}")

    # Drop duplicate rows except for the first occurrence, based on all columns except timestamp and flow_id
    id_columns = []
    if timestamp_col:
        id_columns.append(timestamp_col)
    if flow_id_col:
        id_columns.append(flow_id_col)

    if len(id_columns) == 0:
        df.drop_duplicates(keep="first", inplace=True)
    else:
        df.drop_duplicates(subset=list(set(
            df.columns) - set(id_columns)), keep="first", inplace=True)
    print(f"==>> after drop_duplicates df.shape[0]: {df.shape[0]}")

    return df


def convert_file(input_path, output_format):
    # Supported output formats
    supported_formats = ['csv', 'parquet', 'pkl']

    # Validate the output format
    if output_format not in supported_formats:
        raise ValueError(
            f"Unsupported output file format. Supported formats are: {', '.join(supported_formats)}")

    # Determine the input format based on the file extension
    input_extension = os.path.splitext(input_path)[1].lower()

    # Read the input file
    if input_extension == '.csv':
        df = pd.read_csv(input_path)
    elif input_extension == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_extension == '.pkl':
        df = pd.read_pickle(input_path)
    else:
        raise ValueError("Unsupported input file format")

    # Determine the output path
    output_path = os.path.splitext(input_path)[0] + f".{output_format}"

    # Save the file in the desired format
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif output_format == 'pkl':
        df.to_pickle(output_path)

    print(f"File saved as {output_path}")


def one_dataset_class_num_col(df, class_num_col, class_col):
    classes = df[class_col].unique()
    label_encoder = LabelEncoder()

    label_encoder.fit(list(classes))
    df[class_num_col] = label_encoder.transform(df[class_col])

    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    print(f"==>> labels_names: {labels_names}")

    return df, labels_names, classes


def two_dataset_class_num_col(df1, df2, class_num_col, class_col, class_num_col2=None, class_col2=None):
    if class_num_col2 == None:
        class_num_col2 = class_num_col
    if class_col2 == None:
        class_col2 = class_col

    classes1 = df1[class_col].unique()
    classes2 = df2[class_col2].unique()

    classes = set(np.concatenate([classes2, classes1]))
    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes))

    df1[class_num_col] = label_encoder.transform(
        df1[class_col])
    df2[class_num_col2] = label_encoder.transform(
        df2[class_col2])
    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    print(f"==>> labels_names: {labels_names}")

    return df1, df2, labels_names


def undersample_classes(df, class_col, n_undersample, fraction=0.5):
    """
    Undersamples the classes with the highest number of records.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        class_col (str): The name of the class column.
        n_undersample (int): The number of classes to undersample.
        fraction (float): The fraction of samples to keep from the undersampled classes.

    Returns:
        pd.DataFrame: The undersampled DataFrame.
    """
    # Group by the class column and get the count of records in each class
    class_counts = df.groupby(class_col).size()

    # Sort the counts in descending order
    class_counts_sorted = class_counts.sort_values(ascending=False)

    # Get the classes with the highest number of records to undersample
    classes_to_undersample = class_counts_sorted.index[:n_undersample]

    # Undersample the classes with the highest number of records
    dfs = []
    for class_label in class_counts_sorted.index:
        print(f"==>> class_label: {class_label}")
        class_df = df[df[class_col] == class_label]
        if class_label in classes_to_undersample:
            # Specify the fraction of samples to keep
            undersampled_df = class_df.sample(frac=fraction)
            dfs.append(undersampled_df)
        else:
            dfs.append(class_df)

    # Concatenate all DataFrames and shuffle the undersampled DataFrame
    df_undersampled = pd.concat(dfs).sample(frac=1).reset_index(drop=True)

    return df_undersampled


# features_analysis.py
def prepare_dataset(original_df, drop_columns, label_col):
    cleaned_df = original_df.drop(drop_columns, axis=1)
    cleaned_df = cleaned_df.drop(label_col, axis=1)
    return cleaned_df


def apply_variance_threshold(df, threshold):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    selected_features = df.columns[selector.get_support(indices=True)]
    dropped_features = [
        col for col in df.columns if col not in selected_features]
    df_filtered = df[selected_features]
    return df_filtered, dropped_features


def apply_correlation_threshold(df, threshold):
    corr_matrix = df.corr()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features = [
        column for column in upper.columns if any(upper[column] > threshold)]
    features_to_remove = set()
    for feature in correlated_features:
        correlated_with_feature = list(upper.index[upper[feature] > threshold])
        for correlated_feature in correlated_with_feature:
            if correlated_feature not in features_to_remove:
                features_to_remove.add(correlated_feature)
    df_filtered = df.drop(features_to_remove, axis=1)
    return df_filtered, features_to_remove


def feature_analysis_pipeline(df, drop_columns, label_col, var_threshold=0.00, corr_threshold=0.75):
    new_df = prepare_dataset(df, drop_columns, label_col)
    new_df, var_dropped = apply_variance_threshold(new_df, var_threshold)
    new_df, corr_dropped = apply_correlation_threshold(new_df, corr_threshold)
    return new_df, var_dropped, corr_dropped


# %% [markdown]
# ### General Utilities

# %%
# utils.py

def calculate_fpr_fnr_with_global(cm):
    """
    Calculate FPR and FNR for each class and globally for a multi-class confusion matrix.

    Parameters:
        cm (numpy.ndarray): Confusion matrix of shape (num_classes, num_classes).

    Returns:
        dict: A dictionary containing per-class and global FPR and FNR.
    """
    num_classes = cm.shape[0]
    results = {"per_class": {}, "global": {}}

    # Initialize variables for global calculation
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    # Per-class calculation
    for class_idx in range(num_classes):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Calculate FPR and FNR for this class
        FPR = FP / (FP + TN) if (FP + TN) != 0 else None
        FNR = FN / (TP + FN) if (TP + FN) != 0 else None

        # Store per-class results
        results["per_class"][class_idx] = {"FPR": FPR, "FNR": FNR}

        # Update global counts
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Global calculation
    global_FPR = total_FP / \
        (total_FP + total_TN) if (total_FP + total_TN) != 0 else None
    global_FNR = total_FN / \
        (total_FN + total_TP) if (total_FN + total_TP) != 0 else None

    results["global"]["FPR"] = global_FPR
    results["global"]["FNR"] = global_FNR

    return results


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalized=False,
                          file_path=None,
                          show_figure=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalized:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    if file_path:
        plt.savefig(file_path)
    if show_figure:
        plt.show()
    return fig


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if verbose is in kwargs, defaulting to False if not provided
        verbose = kwargs.get("verbose", False)
        if verbose:
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            print(
                f"==>> {func.__name__}: {result}, in {str(timeit.default_timer() - start_time)} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper


# %% [markdown]
# ### Graph Utilities

# %%
# centralities.py

def cal_betweenness_centrality(G):
    G1 = ig.Graph.from_networkx(G)
    estimate = G1.betweenness(directed=True)
    b = dict(zip(G1.vs["_nx_name"], estimate))
    return betweenness_rescale(b, G1.vcount(), True)


def cal_k_core(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    kcore_dict_eachNode = nx.core_number(G)

    kcore_dict_eachNode_normalized = hm_rescale(kcore_dict_eachNode)

    return kcore_dict_eachNode_normalized


def cal_k_truss(G):
    sr_node_ktruss_dict = {}
    n = G.number_of_nodes()
    G = ig.Graph.from_networkx(G)
    ktrussdict = ktruss(G)
    nodetruss = [0] * n
    for edge in G.es:
        source = edge.source
        target = edge.target
        if not (source == target):
            t = ktrussdict[(source, target)]
        else:
            t = 0
        nodetruss[source] = max(nodetruss[source], t)
        nodetruss[target] = max(nodetruss[target], t)
    d = {}
    node_index = 0
    node_truss_value = 0
    while (node_index < len(nodetruss)):
        d[G.vs[node_index]["_nx_name"]] = nodetruss[node_truss_value]
        node_truss_value = node_truss_value+1
        node_index = node_index+1
    # print(d)
    return hm_rescale(d)


def edge_support(G):
    nbrs = dict((v.index, set(G.successors(v)) | set(G.predecessors(v)))
                for v in G.vs)
    support = {}
    for e in G.es:
        nod1, nod2 = e.source, e.target
        nod1_nbrs = set(nbrs[nod1])-set([nod1])
        nod2_nbrs = set(nbrs[nod2])-set([nod2])
        sup = len(nod1_nbrs.intersection(nod2_nbrs))
        support[(nod1, nod2)] = sup
    return support


def ktruss(G):
    support = edge_support(G)
    edges = sorted(support, key=support.get)  # type: ignore
    bin_boundaries = [0]
    curr_support = 0
    for i, e in enumerate(edges):
        if support[e] > curr_support:
            bin_boundaries.extend([i]*(support[e]-curr_support))
            curr_support = support[e]

    edge_pos = dict((e, pos) for pos, e in enumerate(edges))

    truss = {}
    neighbors = G.neighborhood()

    nbrs = dict(
        (v.index, (set(neighbors[v.index])-set([v.index]))) for v in G.vs)

    for e in edges:
        u, v = e[0], e[1]
        if not (u == v):
            common_nbrs = set(nbrs[u]).intersection(nbrs[v])
            for w in common_nbrs:
                if (u, w) in support:
                    e1 = (u, w)
                else:
                    e1 = (w, u)
                if (v, w) in support:
                    e2 = (v, w)
                else:
                    e2 = (w, v)
                pos = edge_pos[e1]
                if support[e1] > support[e]:
                    bin_start = bin_boundaries[support[e1]]
                    edge_pos[e1] = bin_start
                    edge_pos[edges[bin_start]] = pos
                    edges[bin_start], edges[pos] = edges[pos], edges[bin_start]
                    bin_boundaries[support[e1]] += 1

                pos = edge_pos[e2]
                if support[e2] > support[e]:
                    bin_start = bin_boundaries[support[e2]]
                    edge_pos[e2] = bin_start
                    edge_pos[edges[bin_start]] = pos
                    edges[bin_start], edges[pos] = edges[pos], edges[bin_start]
                    bin_boundaries[support[e2]] += 1

                support[e1] = max(support[e], support[e1]-1)
                support[e2] = max(support[e], support[e2]-1)

            truss[e] = support[e] + 2
            if v in nbrs[u]:
                nbrs[u].remove(v)
            if u in nbrs[v]:
                nbrs[v].remove(u)
    return truss


def add_centralities(df, new_path, graph_path, dataset, cn_measures, network_features, G=None, create_using=nx.DiGraph(), communities=None, G1=None, part=None):

    if not G:
        print("constructing graph")
        G = nx.from_pandas_edgelist(
            df, source=dataset.src_ip_col, target=dataset.dst_ip_col, create_using=create_using)
        G.remove_nodes_from(list(nx.isolates(G)))
        for node in G.nodes():
            G.nodes[node]['label'] = node

    need_communities = False
    comm_list = ["local_betweenness", "global_betweenness", "local_degree", "global_degree", "local_eigenvector",
                 "global_eigenvector", "local_closeness", "global_closeness", "local_pagerank", "global_pagerank", "Comm", "mv"]
    if any(value in comm_list for value in cn_measures):
        need_communities = True

    if need_communities and not communities:
        print("calculating communities")
        if not G1:
            G1 = ig.Graph.from_networkx(G)
            labels = [G.nodes[node]['label'] for node in G.nodes()]
            G1.vs['label'] = labels
        if not part:
            part = G1.community_infomap()

        communities = []
        for com in part:
            communities.append([G1.vs[node_index]['label']
                               for node_index in com])
    if communities:
        community_labels = {}
        for i, community in enumerate(communities):
            for node in community:
                community_labels[node] = i

        nx.set_node_attributes(G, community_labels, "new_community")

        intra_graph, inter_graph = separate_graph(G, communities)

    simple_graph = create_using != nx.MultiDiGraph
    if G:
        simple_graph = type(G) is not nx.MultiDiGraph

    if "betweenness" in cn_measures:
        print("calculating betweenness")
        nx.set_node_attributes(G, cal_betweenness_centrality(G), "betweenness")
        print("calculated betweenness")
    if "local_betweenness" in cn_measures:
        print("calculating local_betweenness")
        nx.set_node_attributes(G, cal_betweenness_centrality(
            intra_graph), "local_betweenness")
        print("calculated local_betweenness")
    if "global_betweenness" in cn_measures:
        print("calculating global_betweenness")
        nx.set_node_attributes(G, cal_betweenness_centrality(
            inter_graph), "global_betweenness")
        print("calculated global_betweenness")
    if "degree" in cn_measures:
        print("calculating degree")
        nx.set_node_attributes(G, nx.degree_centrality(G), "degree")
        print("calculated degree")
    if "local_degree" in cn_measures:
        print("calculating local_degree")
        nx.set_node_attributes(
            G, nx.degree_centrality(intra_graph), "local_degree")
        print("calculated local_degree")
    if "global_degree" in cn_measures:
        print("calculating global_degree")
        nx.set_node_attributes(G, nx.degree_centrality(
            inter_graph), "global_degree")
        print("calculated global_degree")
    if "eigenvector" in cn_measures and simple_graph:
        print("calculating eigenvector")
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            G, max_iter=600), "eigenvector")
        print("calculated eigenvector")
    if "local_eigenvector" in cn_measures and simple_graph:
        print("calculating local_eigenvector")
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            intra_graph), "local_eigenvector")
        print("calculated local_eigenvector")
    if "global_eigenvector" in cn_measures and simple_graph:
        print("calculating global_eigenvector")
        nx.set_node_attributes(G, nx.eigenvector_centrality(
            inter_graph), "global_eigenvector")
        print("calculated global_eigenvector")
    if "closeness" in cn_measures:
        print("calculating closeness")
        nx.set_node_attributes(G, nx.closeness_centrality(G), "closeness")
        print("calculated closeness")
    if "local_closeness" in cn_measures:
        print("calculating local_closeness")
        nx.set_node_attributes(G, nx.closeness_centrality(
            intra_graph), "local_closeness")
        print("calculated local_closeness")
    if "global_closeness" in cn_measures:
        print("calculating global_closeness")
        nx.set_node_attributes(G, nx.closeness_centrality(
            inter_graph), "global_closeness")
        print("calculated global_closeness")
    if "pagerank" in cn_measures:
        print("calculating pagerank")
        nx.set_node_attributes(G, nx.pagerank(G, alpha=0.85), "pagerank")
        print("calculated pagerank")
    if "local_pagerank" in cn_measures:
        print("calculating local_pagerank")
        nx.set_node_attributes(G, nx.pagerank(
            intra_graph, alpha=0.85), "local_pagerank")
        print("calculated local_pagerank")
    if "global_pagerank" in cn_measures:
        print("calculating global_pagerank")
        nx.set_node_attributes(G, nx.pagerank(
            inter_graph, alpha=0.85), "global_pagerank")
        print("calculated global_pagerank")
    if "k_core" in cn_measures and simple_graph:
        print("calculating k_core")
        nx.set_node_attributes(G, cal_k_core(G), "k_core")
        print("calculated k_core")
    if "k_truss" in cn_measures:
        print("calculating k_truss")
        nx.set_node_attributes(G, cal_k_truss(G), "k_truss")
        print("calculated k_truss")

    if graph_path:
        nx.write_gexf(G, graph_path)

    features_dicts = {}
    for measure in cn_measures:
        features_dicts[measure] = nx.get_node_attributes(G, measure)
        print(f"==>> features_dicts: {measure , len(features_dicts[measure])}")

    for feature in network_features:
        if feature[:3] == "src":
            df[feature] = df.apply(lambda row: features_dicts[feature[4:]].get(
                row[dataset.src_ip_col], -1), axis=1)
        if feature[:3] == "dst":
            df[feature] = df.apply(lambda row: features_dicts[feature[4:]].get(
                row[dataset.dst_ip_col], -1), axis=1)

    if new_path:
        df.to_parquet(new_path)
        print(f"DataFrame written to {new_path}")

    return network_features


def normalize_centrality(centrality_dict):
    # Extract values and reshape for sklearn
    values = np.array(list(centrality_dict.values())).reshape(-1, 1)

    # Apply z-score normalization
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(values).flatten()

    # Create a dictionary of normalized values
    normalized_centrality = {node: norm_value for node, norm_value in zip(
        centrality_dict.keys(), normalized_values)}

    return normalized_centrality


def add_centralities_as_node_features(df, G, graph_path, dataset, cn_measures, create_using=nx.DiGraph()):

    if not G:
        G = nx.from_pandas_edgelist(
            df, source=dataset.src_ip_col, target=dataset.dst_ip_col, create_using=create_using)

    G.remove_nodes_from(list(nx.isolates(G)))
    for node in G.nodes():
        G.nodes[node]['label'] = node

    compute_communities = False
    comm_list = ["local_betweenness", "global_betweenness", "local_degree", "global_degree", "local_eigenvector",
                 "global_eigenvector", "local_closeness", "global_closeness", "local_pagerank", "global_pagerank", "Comm", "mv"]
    if any(value in comm_list for value in cn_measures):
        compute_communities = True

    if compute_communities:
        G1 = ig.Graph.from_networkx(G)
        labels = [G.nodes[node]['label'] for node in G.nodes()]
        G1.vs['label'] = labels

        part = G1.community_infomap()
        communities = []
        for com in part:
            communities.append([G1.vs[node_index]['label']
                               for node_index in com])

        community_labels = {}
        for i, community in enumerate(communities):
            for node in community:
                community_labels[node] = i

        nx.set_node_attributes(G, community_labels, "new_community")

        intra_graph, inter_graph = separate_graph(G, communities)

    if "betweenness" in cn_measures:
        normalized_centrality = normalize_centrality(
            cal_betweenness_centrality(G))
        nx.set_node_attributes(G, normalized_centrality, "betweenness")
        print("calculated betweenness")
    if "local_betweenness" in cn_measures:
        normalized_centrality = normalize_centrality(
            cal_betweenness_centrality(intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_betweenness")
        print("calculated local_betweenness")
    if "global_betweenness" in cn_measures:
        normalized_centrality = normalize_centrality(
            cal_betweenness_centrality(inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_betweenness")
        print("calculated global_betweenness")
    if "degree" in cn_measures:
        normalized_centrality = normalize_centrality(nx.degree_centrality(G))
        nx.set_node_attributes(G, normalized_centrality, "degree")
        print("calculated degree")
    if "local_degree" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.degree_centrality(intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_degree")
        print("calculated local_degree")
    if "global_degree" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.degree_centrality(inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_degree")
        print("calculated global_degree")
    if "eigenvector" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.eigenvector_centrality(G, max_iter=600))
        nx.set_node_attributes(G, normalized_centrality, "eigenvector")
        print("calculated eigenvector")
    if "local_eigenvector" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.eigenvector_centrality(intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_eigenvector")
        print("calculated local_eigenvector")
    if "global_eigenvector" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.eigenvector_centrality(inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_eigenvector")
        print("calculated global_eigenvector")
    if "closeness" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.closeness_centrality(G))
        nx.set_node_attributes(G, normalized_centrality, "closeness")
        print("calculated closeness")
    if "local_closeness" in cn_measures:
        normalized_centrality = normalize_centrality(nx.closeness_centrality(
            intra_graph))
        nx.set_node_attributes(G, normalized_centrality, "local_closeness")
        print("calculated local_closeness")
    if "global_closeness" in cn_measures:
        normalized_centrality = normalize_centrality(nx.closeness_centrality(
            inter_graph))
        nx.set_node_attributes(G, normalized_centrality, "global_closeness")
        print("calculated global_closeness")
    if "pagerank" in cn_measures:
        normalized_centrality = normalize_centrality(
            nx.pagerank(G, alpha=0.85))
        nx.set_node_attributes(G, normalized_centrality, "pagerank")
        print("calculated pagerank")
    if "local_pagerank" in cn_measures:
        normalized_centrality = normalize_centrality(nx.pagerank(
            intra_graph, alpha=0.85))
        nx.set_node_attributes(G, normalized_centrality, "local_pagerank")
        print("calculated local_pagerank")
    if "global_pagerank" in cn_measures:
        normalized_centrality = normalize_centrality(nx.pagerank(
            inter_graph, alpha=0.85))
        nx.set_node_attributes(G, normalized_centrality, "global_pagerank")
        print("calculated global_pagerank")
    if "k_core" in cn_measures:
        normalized_centrality = normalize_centrality(cal_k_core(G))
        nx.set_node_attributes(G, normalized_centrality, "k_core")
        print("calculated k_core")
    if "k_truss" in cn_measures:
        normalized_centrality = normalize_centrality(cal_k_truss(G))
        nx.set_node_attributes(G, normalized_centrality, "k_truss")
        print("calculated k_truss")

    if graph_path:
        nx.write_gexf(G, graph_path)

    return G


# graph_construction.py

def create_weightless_flow_graph(df, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, new_graph_path=None):

    try:
        # Record the start time
        start = time.time()

        if multi_graph or line_graph:
            base = nx.MultiDiGraph()
        else:
            base = nx.DiGraph()

        # Create the directed graph from the pandas dataframe
        G = nx.from_pandas_edgelist(df,
                                    source=src_ip_col,
                                    target=dst_ip_col,
                                    create_using=base)

        if line_graph:
            G = nx.line_graph(G)

        if new_graph_path:
            # Save the graph to a GEXF file
            nx.write_gexf(G, new_graph_path)

            print(
                f"Graph created and saved to {new_graph_path} in {time.time() - start:.2f} seconds.")

        else:
            print(f"Graph created in {time.time() - start:.2f} seconds.")

        return G
    except Exception as e:
        print(f"An error occurred: {e}")


def define_sessions(df, timestamp_col, src_ip_col, dst_ip_col, src_port_col, dst_port_col, protocol_col=None, timeout=pd.Timedelta(minutes=5)):
    df = df.sort_values(by=timestamp_col)
    sessions = []
    current_session_id = 0
    last_seen = {}

    for index, row in df.iterrows():
        if protocol_col:
            tuples = (row[src_ip_col], row[dst_ip_col],
                      row[src_port_col], row[dst_port_col], row[protocol_col])
        else:
            tuples = (row[src_ip_col], row[dst_ip_col],
                      row[src_port_col], row[dst_port_col])
        if tuples in last_seen:
            if timeout and row[timestamp_col] - last_seen[tuples] > timeout:
                current_session_id += 1
        else:
            current_session_id += 1
        last_seen[tuples] = row[timestamp_col]
        sessions.append(current_session_id)

    df['session_id'] = sessions
    return df


def create_weightless_session_graph(df, src_ip_col, dst_ip_col, multi_graph=False, line_graph=False, folder_path=None, edge_attr=None, file_type="gexf"):
    try:
        # Record the start time
        start_time = time.time()

        graphs = []

        if multi_graph or line_graph:
            base_graph_type = nx.MultiDiGraph
        else:
            base_graph_type = nx.DiGraph

        # Iterate over each session in the DataFrame
        for session_id, df_session in df.groupby('session_id'):
            # Create a graph from the session
            G = nx.from_pandas_edgelist(df_session,
                                        source=src_ip_col,
                                        target=dst_ip_col,
                                        edge_attr=edge_attr,
                                        create_using=base_graph_type())

            if line_graph:
                G_line_graph = nx.line_graph(G)
                G_line_graph.add_nodes_from(
                    (node, G.edges[node]) for node in G_line_graph)
                G = G_line_graph

            if folder_path:
                if file_type == "gexf":
                    filename = os.path.join(
                        folder_path, 'graphs', f'graph_{session_id}.gexf')
                    # Save the graph to a file
                    nx.write_gexf(G, filename)

                if file_type == "pkl":
                    filename = os.path.join(
                        folder_path, 'graphs', f'graph_{session_id}.pkl')

                    # Save the graph to a file
                    with open(filename, "wb") as f:
                        pickle.dump(G, f)

                calculate_graph_measures(
                    G, os.path.join(folder_path, 'graph_measures', f'graph_{session_id}_measures.json'))

            # Append the graph to the list
            graphs.append(G)

        print(f"Graphs created in {time.time() - start_time:.2f} seconds.")

        return graphs

    except Exception as e:
        print(f"An error occurred: {e}")


def create_weightless_window_graph(df, dataset, window_size=20000, cn_measures=None, network_features=None, multi_graph=False, line_graph=False, folder_path=None, test_percentage=None, edge_attr=None, file_type="gexf"):

    try:
        # Record the start time
        start_time = time.time()

        # graphs = []

        # Total number of records
        total_records = len(df)

        if multi_graph or line_graph:
            base = nx.MultiDiGraph()
        else:
            base = nx.DiGraph()

        # Iterate over the DataFrame in chunks
        i = 0
        number_of_groups = math.ceil(total_records / window_size)
        print(f"==>> number_of_groups: {number_of_groups}")

        if test_percentage:
            # folder_path += "_train_test"

            number_of_test_groups = math.ceil(
                number_of_groups * test_percentage / 100)

            number_of_train_groups = number_of_groups - number_of_test_groups
            print(f"==>> number_of_train_groups: {number_of_train_groups}")
            print(f"==>> number_of_test_groups: {number_of_test_groups}")

        for start in range(0, total_records, window_size):

            df_chunk = df.iloc[start:start + window_size]

            # Create a graph from the chunk
            G = nx.from_pandas_edgelist(df_chunk,
                                        source=dataset.src_ip_col,
                                        target=dataset.dst_ip_col,
                                        edge_attr=edge_attr,
                                        create_using=base)
            if cn_measures and network_features:
                add_centralities(df=None, new_path=None, graph_path=None, dataset=dataset, G=G,
                                 cn_measures=cn_measures, network_features=network_features, create_using=nx.MultiDiGraph())
            if line_graph:
                G_line_graph = nx.line_graph(G)
                G_line_graph.add_nodes_from(
                    (node, G.edges[node]) for node in G_line_graph)
                G = G_line_graph
            if folder_path:
                # Ensure the folder path exists
                os.makedirs(folder_path, exist_ok=True)

                if file_type == "gexf":
                    filename = os.path.join(folder_path, f'graph_{i}.gexf')
                    # Save the graph to a file
                    nx.write_gexf(G, filename)

                if file_type == "pkl":
                    if test_percentage:
                        if i < number_of_train_groups:

                            folder_path_train = os.path.join(
                                folder_path, 'training')
                            os.makedirs(folder_path_train, exist_ok=True)
                            filename = os.path.join(
                                folder_path_train, f'graph_{i}.pkl')
                        else:

                            folder_path_test = os.path.join(
                                folder_path, 'testing')
                            os.makedirs(folder_path_test, exist_ok=True)
                            filename = os.path.join(
                                folder_path_test, f'graph_{i}.pkl')
                    else:
                        folder_path_all = os.path.join(
                            folder_path, 'graphs')
                        os.makedirs(folder_path_all, exist_ok=True)
                        filename = os.path.join(
                            folder_path_all, f'graph_{i}.pkl')

                    # Save the graph to a file
                    with open(filename, "wb") as f:
                        pickle.dump(G, f)

                graph_measures = calculate_graph_measures(
                    G, os.path.join(folder_path, f'graph_{i}_measures.json'))
                print(f"==>> graph_measures of graph_{i}: {graph_measures}")

            # Append the graph to the list
            # graphs.append(G)
            i += 1

        print(f"Graph created in {time.time() - start_time:.2f} seconds.")

        # return graphs

    except Exception as e:
        print(f"An error occurred: {e}")


# graph_measures.py

@time_execution
def number_of_nodes(G, verbose):
    return G.number_of_nodes()


@time_execution
def number_of_edges(G, verbose):
    return G.number_of_edges()


@time_execution
def is_strongly_connected(G, verbose):
    return nx.is_strongly_connected(G)


@time_execution
def transitivity(G, verbose):
    return nx.transitivity(G)


@time_execution
def density(G, verbose):
    return nx.density(G)


@time_execution
def mixing_parameter(G, communities, verbose):

    # Step 1: Map each node to its community
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    # Step 2: Count inter-cluster edges efficiently
    inter_cluster_edges = 0
    for u, v in G.edges():
        # Directly check if u and v belong to different communities
        if node_to_community[u] != node_to_community[v]:
            inter_cluster_edges += 1

    mixing_parameter = inter_cluster_edges / G.number_of_edges()

    return mixing_parameter


@time_execution
def modularity(G, communities, verbose):

    start_time = timeit.default_timer()
    modularity = nx.community.modularity(G, communities)
    if verbose:
        print(
            f"==>> modularity: {modularity}, in {str(timeit.default_timer() - start_time)} seconds")

    return modularity


def get_degrees(G, verbose):
    start_time = timeit.default_timer()
    degrees = [degree for _, degree in G.degree()]
    if verbose:
        print(
            f"==>> calculated degrees, in {str(timeit.default_timer() - start_time)} seconds")
    return degrees


@time_execution
def find_communities(G, verbose):

    start_time = timeit.default_timer()
    G1 = ig.Graph.from_networkx(G)

    part = G1.community_infomap()
    # part = G1.community_multilevel()
    # part = G1.community_spinglass()
    # part = G1.community_edge_betweenness()

    communities = []
    for com in part:
        communities.append([G1.vs[node_index]['_nx_name']
                           for node_index in com])

    # communities = nx.community.louvain_communities(G)
    if verbose:
        print(
            f"==>> number_of_communities: {len(communities)}, in {str(timeit.default_timer() - start_time)} seconds")

    return G1, part, communities


def calculate_graph_measures(G, file_path=None, verbose=False, communities=None):

    properties = {}

    properties["number_of_nodes"] = number_of_nodes(G, verbose)
    properties["number_of_edges"] = number_of_edges(G, verbose)

    degrees = get_degrees(G, verbose)

    properties["max_degree"] = max(degrees)
    properties["avg_degree"] = sum(degrees) / len(degrees)

    if type(G) is nx.DiGraph or type(G) is nx.Graph:
        properties["transitivity"] = transitivity(G, verbose)

    properties["density"] = density(G, verbose)

    if communities:
        properties["number_of_communities"] = len(communities)
        properties["mixing_parameter"] = mixing_parameter(
            G, communities, verbose)
        properties["modularity"] = modularity(G, communities, verbose)

    if file_path:
        outfile = open(file_path, 'w')
        outfile.writelines(json.dumps(properties))
        outfile.close()

    return properties


@time_execution
def class_pairs(df, source_ip, destination_ip, class_column, results_dict, verbose, folder_path=None):
    # Initialize lists to store results
    same_class_pairs = {}
    mixed_class_pairs = []

    # Group by source and destination IP addresses
    for (source, destination), group in df.groupby([source_ip, destination_ip]):
        unique_classes = group[class_column].unique()
        if len(unique_classes) == 1:
            # All records have the same class
            class_label = str(unique_classes[0])
            if class_label not in same_class_pairs:
                same_class_pairs[class_label] = []
            same_class_pairs[class_label].append({
                'node_pair': (source, destination),
                'num_instances': len(group)
            })
        else:
            # Mixed class scenario
            class_counts = group[class_column].value_counts().to_dict()
            total_instances = len(group)
            class_percentages = {
                str(cls): count / total_instances for cls, count in class_counts.items()}
            mixed_class_pairs.append({
                'node_pair': (source, destination),
                'class_counts': class_counts,
                'class_percentages': class_percentages
            })

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "same_class_pairs.json"), "w") as f:
            f.writelines(json.dumps(same_class_pairs, cls=NumpyEncoder))

        with open(os.path.join(folder_path, "mixed_class_pairs.json"), "w") as f:
            f.writelines(json.dumps(mixed_class_pairs, cls=NumpyEncoder))

    # Total counts
    total_same_class_pairs = sum(len(pairs)
                                 for pairs in same_class_pairs.values())
    total_mixed_class_pairs = len(mixed_class_pairs)

    if verbose:
        print("\nTotal number of same class pairs:", total_same_class_pairs)
        print("Total number of mixed class pairs:", total_mixed_class_pairs)

    results_dict["total_same_class_pairs"] = total_same_class_pairs
    results_dict["total_mixed_class_pairs"] = total_mixed_class_pairs

    # Interpretation:
    # - `same_class_pairs` contains node pairs with consistent classes across all records, including the number of instances.
    # - `mixed_class_pairs` contains node pairs with mixed classes, the counts and percentages for each class.
    # - Total counts provide an overview of the dataset's class consistency.


@time_execution
def attackers_victims(graph, results_dict, label_col, verbose):
    attackers = set()
    victims = set()

    for u, v, data in graph.edges(data=True):
        if data[label_col] == 1:
            attackers.add(u)
            victims.add(v)

    # Step 2: Count unique attackers and victims
    num_attackers = len(attackers)
    num_victims = len(victims)

    # Step 3: Calculate proportions
    total_nodes = graph.number_of_nodes()
    attacker_proportion = num_attackers / total_nodes if total_nodes > 0 else 0
    victim_proportion = num_victims / total_nodes if total_nodes > 0 else 0

    if verbose:
        print("Number of Attackers:", num_attackers)
        print("Number of Victims:", num_victims)
        print("Proportion of Attackers:", attacker_proportion)
        print("Proportion of Victims:", victim_proportion)

    results_dict["total_nodes"] = total_nodes
    results_dict["Number of Attackers"] = num_attackers
    results_dict["Number of Victims"] = num_victims
    results_dict["Proportion of Attackers"] = attacker_proportion
    results_dict["Proportion of Victims"] = victim_proportion
    results_dict["intersection between attacks and victims"] = len(
        attackers.intersection(victims))

    # Interpretation:
    # - Attackers: Source nodes of edges labeled as "Attack".
    # - Victims: Target nodes of edges labeled as "Attack".
    # - These metrics provide insight into the roles of nodes in attack scenarios.


@time_execution
def cal_clustering_coefficients(graph, results_dict, verbose):

    # Clustering Coefficient Distribution Metric
    # Convert MultiDiGraph to Graph for clustering
    clustering_coefficients = nx.clustering(nx.Graph(graph))
    clustering_values = list(clustering_coefficients.values())
    mean_clustering = np.mean(clustering_values)
    std_clustering = np.std(clustering_values)

    if verbose:
        print("Mean Clustering Coefficient:", mean_clustering)
        print("Standard Deviation of Clustering Coefficients:", std_clustering)

    results_dict["Mean Clustering Coefficient"] = mean_clustering
    results_dict["Standard Deviation of Clustering Coefficients"] = std_clustering


@time_execution
def cal_degree_assortativity(graph, results_dict, verbose):
    # Graph Assortativity Metric
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(graph)
        results_dict["Graph Degree Assortativity Coefficient"] = degree_assortativity
        if verbose:
            print("Degree Assortativity Coefficient:", degree_assortativity)
    except nx.NetworkXError as e:
        results_dict["Graph Degree Assortativity Coefficient"] = "not applicable"
        if verbose:
            print("Error calculating assortativity:", e)


@time_execution
def cal_diameter(graph, results_dict, verbose):
    # Graph Diameter Metric
    try:
        if nx.is_strongly_connected(graph):
            diameter = nx.diameter(graph)
            results_dict["diameter"] = diameter
            if verbose:
                print("Graph Diameter multidigraph:", diameter)
        else:
            results_dict["diameter"] = "not applicable"
            if verbose:
                print("Graph is not strongly connected, diameter is undefined.")

    except nx.NetworkXError as e:
        print("Error calculating diameter:", e)


@time_execution
def path_length_distribution(graph, results_dict, verbose):
    # Path Length Distribution Metric
    try:
        path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
        all_lengths = [length for source in path_lengths.values()
                       for length in source.values()]
        mean_path_length = np.mean(all_lengths)
        std_path_length = np.std(all_lengths)
        if verbose:
            print("Mean Path Length MultiDiGraph:", mean_path_length)
            print("Standard Deviation of Path Lengths MultiDiGraph:", std_path_length)
        results_dict["Mean Path Length"] = mean_path_length
        results_dict["Standard Deviation of Path Lengths"] = std_path_length

    except nx.NetworkXError as e:
        results_dict["Mean Path Length"] = "not applicable"
        results_dict["Standard Deviation of Path Lengths"] = "not applicable"
        if verbose:
            print("Error calculating path length distribution:", e)

    # Interpretation:
    # - Diameter: Longest shortest path in the graph (undefined for disconnected graphs).
    # - Assortativity: Correlation of node degrees (positive, negative, or neutral).
    # - Clustering Coefficients: Measure of local connectivity (distribution provides network structure insights).
    # - Path Lengths: Reachability analysis using shortest paths.


def check_if_scale_free(centrality_sequence, title="Centrality Distribution", verbose=False):
    fit = powerlaw.Fit(centrality_sequence)
    if verbose:
        print(f"==>> fit.alpha: {fit.alpha}")

    bins = np.logspace(np.log10(min(centrality_sequence)),
                       np.log10(max(centrality_sequence)), 20)
    hist, bins = np.histogram(centrality_sequence, bins=bins, density=True)

    # Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot the observed degree distribution (log-log scale)
    plt.scatter(bin_centers, hist, color='blue',
                alpha=0.7, label="Observed Data")

    # Overlay the fitted power-law line
    fit.power_law.plot_pdf(color='red', linestyle="--",
                           label=f"Power-Law Fit (γ={fit.alpha:.2f})")

    # Log scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Labels, title, and legend
    plt.xlabel("Centrality Value (k)")
    plt.ylabel("P(k) (Probability Density)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.show()

    scale_free = fit.alpha > 2 and fit.alpha < 3
    if verbose:
        if scale_free:
            print("This graph is likely scale-free.")
        else:
            print("This graph is NOT scale-free.")

    return fit.alpha, scale_free


@time_execution
def centrality_analysis(centrality_sequence, dataset_name, centrality_name, verbose):
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(
        centrality_sequence.reshape(-1, 1)).flatten()

    centrality_skewness = skew(normalized_values)

    vc = np.unique(normalized_values, return_counts=True)[1]
    centrality_entropy = entropy(pk=vc)

    alpha, scale_free = check_if_scale_free(
        centrality_sequence, title=f"{centrality_name} Distribution of :{dataset_name}", verbose=verbose)

    return centrality_skewness, centrality_entropy, alpha, scale_free


@time_execution
def compute_edge_class_entropy(graph, label_col, verbose):
    """
    Computes the average entropy of edge class distributions across all nodes.

    Args:
        graph: A NetworkX MultiGraph or MultiDiGraph.
        label_col: The key in edge attributes that stores the class label.

    Returns:
        avg_entropy: The average entropy of all nodes.
    """
    node_entropy = {}

    for node in graph.nodes():
        edge_class_counts = defaultdict(int)
        total_edges = 0

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)

            # If graph is a MultiGraph, edge_data is a dictionary with multiple edges
            if isinstance(edge_data, dict):
                for edge_id, data in edge_data.items():
                    edge_class = data.get(label_col, None)
                    if edge_class is not None:
                        edge_class_counts[edge_class] += 1
                        total_edges += 1
            else:  # If it's a simple Graph/DiGraph, just process normally
                edge_class = edge_data.get(label_col, None)
                if edge_class is not None:
                    edge_class_counts[edge_class] += 1
                    total_edges += 1

        # Compute entropy for the node
        if total_edges > 0:
            probs = np.array(list(edge_class_counts.values())) / total_edges
            entropy = -np.sum(probs * np.log2(probs))  # Shannon entropy
        else:
            entropy = 0  # If node has no edges, entropy is 0

        node_entropy[node] = entropy

    # Compute average entropy across all nodes
    avg_entropy = np.mean(list(node_entropy.values())) if node_entropy else 0
    return avg_entropy

    # Higher entropy → The node interacts with diverse attack/traffic types.
    # Lower entropy → The node interacts with one dominant edge class.
    # Zero entropy → Either an isolated node or all edges belong to the same class.


@time_execution
def compute_avg_edge_class_diversity(graph, label_col, verbose):
    """
    Computes the average number of unique edge classes per node.

    Args:
        graph: A NetworkX MultiGraph or MultiDiGraph.
        label_col: The key in edge attributes that stores the class label.

    Returns:
        avg_diversity: Average unique edge class count per node.
    """
    node_diversity = []

    for node in graph.nodes():
        unique_classes = set()
        total_edges = 0

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)

            # If MultiGraph, iterate through multiple edges
            if isinstance(edge_data, dict):
                for edge_id, data in edge_data.items():
                    edge_class = data.get(label_col, None)
                    if edge_class is not None:
                        unique_classes.add(edge_class)
                        total_edges += 1
            else:  # If it's a simple Graph, process normally
                edge_class = edge_data.get(label_col, None)
                if edge_class is not None:
                    unique_classes.add(edge_class)
                    total_edges += 1

        # Compute diversity for the node
        if total_edges > 0:
            diversity = len(unique_classes) / total_edges
            node_diversity.append(diversity)

    return np.mean(node_diversity) if node_diversity else 0

    # Higher diversity → The node interacts with multiple types of traffic.
    # Lower diversity → The node interacts mostly with one edge class.
    # Zero diversity → The node has no edges or only interacts with one type.


# graph_utils.py
def betweenness_rescale(betweenness, n, normalized, directed=False, k=None, endpoints=False):
    if normalized:
        if endpoints:
            if n < 2:
                scale = None  # no normalization
            else:
                # Scale factor should include endpoint nodes
                scale = 1 / (n * (n - 1))
        elif n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def hm_rescale(dict):
    max_list = []
    for i in dict.values():
        max_list.append(i)
    # Rescaling

    def max_num_in_list(list):
        max = list[0]
        for a in list:
            if a > max:
                max = a
        return max

        # get the factor to divide by max
    max_factor = max_num_in_list(max_list)
    x = {}
    for key, value in dict.items():
        x[key] = value / max_factor
    return x


def separate_graph(graph, communities):
    """
    Separates a graph into intra-community and inter-community edges.

    Parameters:
    - graph: A NetworkX graph
    - communities: A list of sets, where each set contains the nodes in a community

    Returns:
    - intra_graph: A graph containing only intra-community edges
    - inter_graph: A graph containing only inter-community edges
    """
    # Create new graphs for intra-community and inter-community edges
    intra_graph = nx.Graph()
    inter_graph = nx.Graph()

    # Add all nodes to both graphs to ensure structure is maintained
    intra_graph.add_nodes_from(graph.nodes())
    inter_graph.add_nodes_from(graph.nodes())

    # Organize communities in a way that allows quick lookup of node to community mapping
    node_to_community = {}
    for community_index, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_index

    # Iterate through each edge in the original graph
    for edge in graph.edges():
        node_u, node_v = edge

        # Determine if an edge is intra-community or inter-community
        if node_to_community.get(node_u) == node_to_community.get(node_v):
            # Intra-community edge
            intra_graph.add_edge(node_u, node_v)
        else:
            # Inter-community edge
            inter_graph.add_edge(node_u, node_v)

    return intra_graph, inter_graph



# %% [markdown]
# ### Loss Functions

# %%
# loss_functions.py

# ---------- shared helpers ----------
def class_counts_from_tensor(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.bincount(labels, minlength=num_classes).clamp_min(1)

def class_balanced_weights_from_counts(counts: torch.Tensor, beta: float = 0.999, normalize: bool = True) -> torch.Tensor:
    # Effective number: (1 - beta^n_c)
    eff_num = 1.0 - torch.pow(torch.tensor(beta, device=counts.device, dtype=torch.float32), counts.to(torch.float32))
    w = (1.0 - beta) / eff_num
    if normalize:
        w = w * (counts.numel() / w.sum())
    return w

# ---------- Loss 1: Class-Balanced (Effective Number) + CE ----------
class ClassBalancedCELoss(nn.Module):
    def __init__(self, counts: torch.Tensor, beta: float = 0.999, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("counts", counts.to(torch.long))
        self.beta = beta
        self.reduction = reduction
        self._refresh()

    def _refresh(self):
        self.register_buffer("weights", class_balanced_weights_from_counts(self.counts, beta=self.beta, normalize=True))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights, reduction=self.reduction)

# ---------- Loss 2: Focal Loss (multiclass; α can be scalar or vector) ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')     # -log p_t
        pt = torch.exp(-ce).clamp(min=1e-12, max=1-1e-12)          # p_t

        if self.alpha is None:
            alpha_t = torch.ones_like(ce)
        elif self.alpha.ndim == 0:
            alpha_t = torch.full_like(ce, float(self.alpha.item()))
        else:
            alpha_t = self.alpha.to(logits.device)[targets]

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

# ---------- Loss 3: LDAM + DRW ----------
class LDAMDRWLoss(nn.Module):
    """
    LDAM margin with optional DRW schedule:
      - subtract m_c from the target logit (m_c = C_margin / n_c^(1/4))
      - before drw_start: plain LDAM-CE
      - after  drw_start: LDAM-CE with class-balanced weights (effective number)
    Call `set_epoch(e)` once per epoch.
    """
    def __init__(self, counts: torch.Tensor, C_margin: float = 0.5, drw_start: int = 10, cb_beta: float = 0.999, reduction: str = "mean"):
        super().__init__()
        counts = counts.to(torch.long)
        self.register_buffer("counts", counts)
        self.register_buffer("m_c", (C_margin / (counts.to(torch.float32).pow(0.25))))
        self.drw_start = drw_start
        self.cb_beta = cb_beta
        self.reduction = reduction
        self.epoch = 0
        # warmup weights (None)
        self.register_buffer("weights", None)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.epoch >= self.drw_start:
            w = class_balanced_weights_from_counts(self.counts, beta=self.cb_beta, normalize=True)
            self.register_buffer("weights", w)
        else:
            self.register_buffer("weights", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # subtract margin on target logit
        idx = torch.arange(logits.size(0), device=logits.device)
        logits_adj = logits.clone()
        logits_adj[idx, targets] = logits_adj[idx, targets] - self.m_c[targets].to(logits.device)

        return F.cross_entropy(logits_adj, targets, weight=self.weights, reduction=self.reduction)

# ---------- Loss 4: Logit-Adjusted CE (priors) ----------
class LogitAdjustedCELoss(nn.Module):
    def __init__(self, counts: torch.Tensor, tau: float = 1.0, reduction: str = "mean"):
        super().__init__()
        priors = (counts.to(torch.float32) / counts.sum()).clamp_(min=1e-12)
        self.register_buffer("adj", tau * priors.log())
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adj.to(logits.device), targets, reduction=self.reduction)

# ---------- Loss 5: Balanced Softmax ----------
class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("adj", counts.to(torch.float32).clamp_min(1).log())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adj.to(logits.device), targets, reduction="mean")

# ---------- Loss factory ----------
def build_imbalance_loss(loss_name: str,
                         num_classes: int,
                         counts: torch.Tensor | None = None,
                         focal_gamma: float = 2.0,
                         focal_alpha=None,
                         cb_beta: float = 0.999,
                         ldam_C_margin: float = 0.5,
                         drw_start: int = 10,
                         cb_beta_drw: float = 0.999,
                         logit_adj_tau: float = 1.0):
    """
    Returns (criterion, needs_epoch_hook)
    """
    needs_counts = loss_name in {"ce_cb","ldam_drw","logit_adj","balanced_softmax"}
    if needs_counts and counts is None:
        raise ValueError(f"{loss_name} requires class counts (tensor shape [C]).")

    if loss_name == "vanilla_ce":
        return nn.CrossEntropyLoss(), False
    if loss_name == "ce_cb":
        return ClassBalancedCELoss(counts=counts, beta=cb_beta), False
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha), False
    if loss_name == "ldam_drw":
        return LDAMDRWLoss(counts=counts, C_margin=ldam_C_margin, drw_start=drw_start, cb_beta=cb_beta_drw), True
    if loss_name == "logit_adj":
        return LogitAdjustedCELoss(counts=counts, tau=logit_adj_tau), False
    if loss_name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts=counts), False
    raise ValueError(f"Unknown loss_name: {loss_name}")

# %% [markdown]
# ### Lightning Data Module

# %%
# lightning_data.py

# Customize this dataset class to encapsulate your graph-loading and processing.


class CustomGraphDataset(Dataset):
    def __init__(self,
                 graphs_folder,
                 split='train',
                 use_node_features=True,
                 multi_class=False,
                 using_masking=False,
                 masked_class=-1,
                 label_col="Label",
                 class_num_col="Class",
                 num_workers=0,
                 device='cpu'):
        """
        Args:
            graphs_folder (str): Path to the folder containing your pkl files.
            split (str): Which split to load; should be one of 'train', 'val', or 'test'.
            use_node_features (bool): Whether to load node features.
            multi_class (bool): Whether you are in multi-class mode.
            using_masking (bool): Whether you want to perform masking.
            masked_class (int): The class index to mask (if applicable).
            device (str): Device on which to place your DGL graphs.
        """
        self.graphs_folder = graphs_folder
        self.split = split
        self.use_node_features = use_node_features
        self.multi_class = multi_class
        self.using_masking = using_masking
        self.masked_class = masked_class
        self.label_col = label_col
        self.class_num_col = class_num_col
        self.num_workers = num_workers
        self.device = device

        # Load and process the graph when the dataset is instantiated.
        self.graph = self._load_and_process_graph()

        # (Optional) Compute class weights once if needed. In this example we compute them
        # using the edge attribute names "label" or "class_num". Adjust as necessary.
        self.class_weights = self._compute_class_weights()

    def _load_and_process_graph(self):
        # Determine the filename based on the split
        filename = os.path.join(self.graphs_folder, f"{self.split}_graph.pkl")
        with open(filename, "rb") as f:
            G_nx = pickle.load(f)  # This is your NetworkX graph

        # Define which edge attributes you need.
        # (You might want to pass these in as parameters instead.)
        edge_attributes = ['h', self.label_col, self.class_num_col]

        # Convert the NetworkX graph to a DGL graph, including node features if desired.
        if self.use_node_features:
            G = from_networkx(G_nx, edge_attrs=edge_attributes,
                              node_attrs=["n_feats"]).to(self.device)
        else:
            G = from_networkx(G_nx, edge_attrs=edge_attributes).to(self.device)

        # Get the number of features from the edge feature "h"
        num_features = G.edata['h'].shape[1]

        # If masking is used in training, filter out the masked edges.
        if self.using_masking and self.split == 'training':
            # Assumes you want to mask based on the "class_num" attribute.
            training_mask = G.edata[self.class_num_col] != self.masked_class
            G = dgl.edge_subgraph(G, training_mask)

        # Process node features:
        if self.use_node_features:
            # For example, you might want to use the "n_feats" field.
            G.ndata["h"] = G.ndata["n_feats"].to(self.device)
        else:
            # Otherwise, initialize node features as ones.
            G.ndata['h'] = torch.ones(
                G.num_nodes(), num_features, device=self.device)

        # Reshape node and edge features if required by your model.
        G.ndata['h'] = G.ndata['h'].reshape(G.ndata['h'].shape[0], 1, -1)
        G.edata['h'] = G.edata['h'].reshape(G.edata['h'].shape[0], 1, -1)

        # Create masks in the edge data for later usage in training/validation/testing.
        if self.split == 'training':
            G.edata['train_mask'] = torch.ones(
                G.edata['h'].shape[0], dtype=torch.bool, device=self.device)
        elif self.split == 'validation':
            G.edata['val_mask'] = torch.ones(
                G.edata['h'].shape[0], dtype=torch.bool, device=self.device)
        elif self.split == 'testing':
            G.edata['test_mask'] = torch.ones(
                G.edata['h'].shape[0], dtype=torch.bool, device=self.device)

        return G

    def _compute_class_weights(self):
        # Compute class weights using sklearn's compute_class_weight.
        # Use the appropriate edge attribute based on multi_class mode.
        if self.multi_class:
            target = self.graph.edata[self.class_num_col].cpu().numpy()
        else:
            target = self.graph.edata[self.label_col].cpu().numpy()
        classes = np.unique(target)
        weights = class_weight.compute_class_weight(
            'balanced', classes=classes, y=target)
        if self.using_masking:
            # If masking is used, you might want to insert a zero weight for the masked class.
            weights = np.insert(weights, self.masked_class, 0)
        return torch.FloatTensor(weights)

    def __len__(self):
        # If you are loading one complete graph per split, the dataset length is 1.
        # If you have many graphs stored in a single file, you would change this.
        return 1

    def __getitem__(self, idx):
        # Since there is only one graph, simply return it.
        # (For a dataset with multiple graphs, return the idx-th graph.)
        return self.graph

# Now create a LightningDataModule that wraps your dataset for train, val, and test.


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, graphs_folder, batch_size=1, **dataset_kwargs):
        """
        Args:
            graphs_folder (str): Directory where your pkl files are stored.
            batch_size (int): Batch size for the DataLoader.
                (If your dataset returns one giant graph per split, batch_size should usually be 1.)
            dataset_kwargs: Additional keyword arguments passed to CustomGraphDataset,
                such as use_node_features, multi_class, using_masking, masked_class, device, etc.
        """
        super().__init__()
        self.graphs_folder = graphs_folder
        self.batch_size = batch_size
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage=None):
        # For the 'fit' stage, load train and validation splits.
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomGraphDataset(
                self.graphs_folder, split='training', **self.dataset_kwargs)
            self.val_dataset = CustomGraphDataset(
                self.graphs_folder, split='validation', **self.dataset_kwargs)
        # For the 'test' stage, load the test split.
        if stage == 'test' or stage is None:
            self.test_dataset = CustomGraphDataset(
                self.graphs_folder, split='testing', **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.dataset_kwargs["num_workers"],
                          collate_fn=lambda x: x[0])

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.dataset_kwargs["num_workers"],
                          collate_fn=lambda x: x[0])

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.dataset_kwargs["num_workers"],
                          collate_fn=lambda x: x[0])


# %% [markdown]
# ### Lightning Model

# %%
# lightning_model.py

def calculate_fpr_fnr_with_global(cm):
    """
    Calculate FPR and FNR for each class and globally for a multi-class confusion matrix.

    Parameters:
        cm (numpy.ndarray): Confusion matrix of shape (num_classes, num_classes).

    Returns:
        dict: A dictionary containing per-class and global FPR and FNR.
    """
    num_classes = cm.shape[0]
    results = {"per_class": {}, "global": {}}

    # Initialize variables for global calculation
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    # Per-class calculation
    for class_idx in range(num_classes):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Calculate FPR and FNR for this class
        FPR = FP / (FP + TN) if (FP + TN) != 0 else None
        FNR = FN / (TP + FN) if (TP + FN) != 0 else None

        # Store per-class results
        results["per_class"][class_idx] = {"FPR": FPR, "FNR": FNR}

        # Update global counts
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Global calculation
    global_FPR = total_FP / \
        (total_FP + total_TN) if (total_FP + total_TN) != 0 else None
    global_FNR = total_FN / \
        (total_FN + total_TP) if (total_FN + total_TP) != 0 else None

    results["global"]["FPR"] = global_FPR
    results["global"]["FNR"] = global_FNR

    return results


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalized=False,
                          file_path=None,
                          show_figure=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalized:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    if file_path:
        plt.savefig(file_path)
    if show_figure:
        plt.show()
    return fig


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class GraphModel(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, config, model_name, labels_mapping, weight_decay=0, using_wandb=False, norm=False, multi_class=False, label_col="Label", class_num_col="Class", batch_size=1):
        """
        Args:
            model: Your graph neural network model (e.g. created via create_model(...))
            criterion: Loss function (e.g. nn.CrossEntropyLoss with your class weights)
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay (L2 regularization) parameter.
            norm (bool): If True, apply edge weight normalization (using EdgeWeightNorm).
            multi_class (bool): If True, use the 'class_num' field in edge data; otherwise, 'label'.
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.labels = list(labels_mapping.values())
        self.labels_mapping = labels_mapping
        self.weight_decay = weight_decay
        self.using_wandb = using_wandb
        self.norm = norm
        self.multi_class = multi_class
        self.label_col = label_col
        self.class_num_col = class_num_col
        self.batch_size = batch_size
        self.save_hyperparameters(config)
        self.train_epoch_metrics = {}
        self.val_epoch_metrics = {}
        self.test_outputs = []
        self.test_prefix = ""

    def forward(self, graph, node_features, edge_features):
        """
        Forward pass on the graph.

        If edge normalization is enabled, we compute and attach normalized edge weights.
        """
        if self.norm:
            # Compute edge normalization weights (assuming 'both' normalization)
            edge_weight = torch.ones(
                graph.num_edges(), dtype=torch.float32, device=graph.device)
            norm_func = EdgeWeightNorm(norm='both')
            norm_edge_weight = norm_func(graph, edge_weight)
            graph.edata['norm_weight'] = norm_edge_weight
        return self.model(graph, node_features, edge_features)

    def training_step(self, batch, batch_idx):
        """
        Training step:
          - Extract node and edge features.
          - Use the training mask to select edges/labels.
          - Compute the loss and accuracy.
        """
        graph = batch
        node_features = graph.ndata['h']
        edge_features = graph.edata['h']
        train_mask = graph.edata['train_mask']
        edge_labels = graph.edata[self.class_num_col] if self.multi_class else graph.edata[self.label_col]

        # Forward pass (apply the model to the entire graph)
        pred = self.forward(graph, node_features, edge_features)
        loss = self.criterion(pred[train_mask], edge_labels[train_mask])

        preds = pred[train_mask].argmax(dim=1).detach().cpu()
        targets = edge_labels[train_mask].detach().cpu()

        train_acc = (preds == targets).float().mean().item() * 100

        weighted_f1 = f1_score(targets, preds, average="weighted") * 100

        self.train_epoch_metrics = {
            "train_loss": loss,
            "train_acc": train_acc,
            "train_f1_score": weighted_f1,
        }
        # Log the metrics (they will appear in the progress bar and logs)
        self.log("train_loss", loss, on_epoch=True, on_step=False, logger=False,
                 prog_bar=True, batch_size=self.batch_size)
        self.log("train_acc", train_acc, on_epoch=True, on_step=False, logger=False,
                 prog_bar=True, batch_size=self.batch_size)
        self.log("train_f1_score", weighted_f1, on_epoch=True, on_step=False, logger=False,
                 prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step:
          - Uses the 'val_mask' and (optionally) node features stored under 'feature'
            if available (otherwise falls back to 'h').
        """
        graph = batch
        # Use node features from 'feature' if available (they may have been preprocessed differently)
        node_features = graph.ndata.get('feature', graph.ndata['h'])
        edge_features = graph.edata['h']
        val_mask = graph.edata['val_mask']
        edge_labels = graph.edata[self.class_num_col] if self.multi_class else graph.edata[self.label_col]

        pred = self.forward(graph, node_features, edge_features)
        loss = self.criterion(pred[val_mask], edge_labels[val_mask]).item()

        preds = pred[val_mask].argmax(dim=1).detach().cpu()
        targets = edge_labels[val_mask].detach().cpu()

        val_acc = (preds == targets).float().mean().item() * 100

        weighted_f1 = f1_score(targets, preds, average="weighted") * 100

        self.val_epoch_metrics = {
            "val_loss": loss,
            "val_acc": val_acc,
            "val_f1_score": weighted_f1,
        }
        self.log("val_loss", loss, on_epoch=True, on_step=False, logger=False,
                 prog_bar=True, batch_size=self.batch_size)
        self.log("val_acc", val_acc, on_epoch=True, on_step=False, logger=False,
                 prog_bar=True, batch_size=self.batch_size)
        self.log("val_f1_score", weighted_f1, on_epoch=True, on_step=False, logger=False,
                 prog_bar=True, batch_size=self.batch_size)
        return {"val_loss": loss, "val_acc": val_acc, "val_f1_score": weighted_f1}

    def on_validation_epoch_end(self):
        epoch_metrics = {}
        epoch_metrics.update(self.train_epoch_metrics)
        epoch_metrics.update(self.val_epoch_metrics)
        self.log_dict(epoch_metrics, on_step=False,
                      on_epoch=True, batch_size=self.batch_size)
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        """
        Test step:
          - Similar to the validation step but using 'test_mask'.
        """
        graph = batch
        node_features = graph.ndata.get('feature', graph.ndata['h'])
        edge_features = graph.edata['h']
        test_mask = graph.edata['test_mask']
        edge_labels = graph.edata[self.class_num_col] if self.multi_class else graph.edata[self.label_col]

        pred = self.forward(graph, node_features, edge_features)
        loss = self.criterion(pred[test_mask], edge_labels[test_mask])

        preds = pred[test_mask].argmax(dim=1).detach().cpu()
        targets = edge_labels[test_mask].detach().cpu()

        test_acc = (preds == targets).float().mean().item() * 100

        weighted_f1 = f1_score(targets, preds, average="weighted") * 100

        self.test_outputs.append(
            {"preds": preds.numpy().tolist(), "targets": targets.numpy().tolist()})

        self.log(f"{self.test_prefix}_test_loss", loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log(f"{self.test_prefix}_test_acc", test_acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log(f"{self.test_prefix}_test_f1", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        return {"test_loss": loss, "test_acc": test_acc, "test_f1": weighted_f1}
        # return {"preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        """
        Aggregate test outputs, compute the confusion matrix and other metrics,
        then log or save the CM plot in the log directory.
        """
        all_preds = []
        all_targets = []
        for output in self.test_outputs:
            all_preds.extend(output["preds"])
            all_targets.extend(output["targets"])

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # log_dir = self.logger.log_dir if hasattr(
        #     self.logger, "log_dir") else "."

        all_targets = np.vectorize(self.labels_mapping.get)(all_targets)
        all_preds = np.vectorize(self.labels_mapping.get)(all_preds)

        # Compute the confusion matrix and classification report.
        cm = confusion_matrix(all_targets, all_preds, labels=self.labels)

        cr = classification_report(
            all_targets, all_preds, digits=4, output_dict=True, zero_division=0)
        report = classification_report(
            all_targets, all_preds, digits=4, output_dict=False, zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100

        results_fpr_fnr = calculate_fpr_fnr_with_global(cm)
        fpr = results_fpr_fnr["global"]["FPR"]
        fnr = results_fpr_fnr["global"]["FNR"]

        # Log scalar metrics (using log_dict is useful for multiple scalars)
        # self.log_dict({
        #     "test_weighted_f1": weighted_f1,
        #     "test_fpr": fpr if fpr is not None else float('nan'),
        #     "test_fnr": fnr if fnr is not None else float('nan')
        # }, prog_bar=True)

        results = {
            "test_weighted_f1": weighted_f1,
            "test_fpr": fpr,
            "test_fnr": fnr,
            "classification_report": cr,
            "results_fpr_fnr": results_fpr_fnr
        }

        os.makedirs("temp", exist_ok=True)
        json_path = os.path.join("temp", f"{self.model_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        if self.using_wandb:
            wandb.save(json_path)

        print("=== Test Evaluation Metrics ===")
        print("Classification Report:\n", report)

        # confusion_matrices = os.path.join(log_dir, "confusion_matrices")
        # cm_path = os.path.join(confusion_matrices, f"{self.model_name}.png")
        # os.makedirs(confusion_matrices, exist_ok=True)

        cm_normalized = confusion_matrix(
            all_targets, all_preds, labels=self.labels, normalize="true")
        fig = plot_confusion_matrix(cm=cm,
                                    normalized=False,
                                    target_names=self.labels,
                                    title=f"Confusion Matrix of {self.model_name}",
                                    file_path=None,
                                    show_figure=False)

        if self.using_wandb:
            wandb.log({f"confusion_matrix_{self.model_name}": wandb.Image(
                fig), "epoch": self.current_epoch})
        fig = plot_confusion_matrix(cm=cm_normalized,
                                    normalized=True,
                                    target_names=self.labels,
                                    title=f"Confusion Matrix of {self.model_name}",
                                    file_path=None,
                                    show_figure=False)
        if self.using_wandb:
            wandb.log({f"confusion_matrix_{self.model_name}_normalized": wandb.Image(
                fig), "epoch": self.current_epoch})

        return {"test_f1": weighted_f1}

    def configure_optimizers(self):
        """
        Configure the optimizer. In this example, we use Adam.
        """
        optimizer = torch.optim.Adam(self.model.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        return optimizer


# %% [markdown]
# ### Model Definitions

# %%
# models.py
# MLPPredictor
class MLPPredictor(nn.Module):
    def __init__(self, in_features, edim, out_classes, activation, residual):
        super().__init__()
        self.residual = residual
        self.activation = activation
        if residual:
            self.W = nn.Linear(in_features * 2 + edim, out_classes)
        else:
            self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']

        h_v = edges.dst['h']
        if self.residual:
            h_uv = edges.data['h']
            h_uv = h_uv.view(h_uv.shape[0], h_uv.shape[2])
            score = self.W(torch.cat([h_u, h_v, h_uv], 1))
        else:
            score = self.W(torch.cat([h_u, h_v], 1))

        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


#############################
#############################
#############################
# E_GCN
class GCNLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, norm=True):
        super(GCNLayer, self).__init__()
        self.W_apply = nn.Linear(ndim_in + edim, ndim_out)
        self.activation = activation
        self.norm = norm

    def message_func(self, edges):
        message = edges.data['h']
        if self.norm:
            norm_weight = edges.data['norm_weight'].unsqueeze(-1).unsqueeze(-1)
            message = norm_weight * message
        return {'m': message}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl

            g.ndata['h'] = nfeats
            g.edata['h'] = efeats

            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            g.ndata['h'] = self.activation(self.W_apply(
                torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class GCN(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout, norm):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    GCNLayer(ndim_in, edim, ndim_out[layer], activation, norm=norm))
            else:
                self.layers.append(
                    GCNLayer(ndim_out[layer-1], edim, ndim_out[layer], activation, norm=norm))

        self.dropout = nn.Dropout(p=dropout)
        self.norm = norm

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class EGCN(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=True, num_class=2, norm=False):
        super().__init__()
        self.gnn = GCN(ndim_in, edim, ndim_out,
                       num_layers, activation, dropout, norm=norm)
        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)


#############################
#############################
#############################
# E_GraphSAGE Model


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, aggregation, num_neighbors=None):
        super(SAGELayer, self).__init__()
        self.W_apply = nn.Linear(ndim_in + edim, ndim_out)
        self.activation = activation
        self.aggregation = aggregation
        self.num_neighbors = num_neighbors

        if aggregation == "pool":
            self.pool_fc = nn.Linear(ndim_out, ndim_out)
        elif aggregation == "lstm":
            self.lstm = nn.LSTM(ndim_out, ndim_out, batch_first=True)

    def message_func(self, edges):
        # if multi_graph then the node features of the source node are repeated
        # after concatenation, for each edge, we have [src_nfeats_1 , ... , src_nfeats_n, efeats_1, ... efeats_m]
        # after that we apply linear layer to create new featurescset called m.
        return {'m': edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl

            # Neighbor sampling
            if self.num_neighbors:
                g = dgl.sampling.sample_neighbors(
                    g, g.nodes(), self.num_neighbors)

                g.ndata['h'] = nfeats
                g.edata['h'] = efeats[g.edata[dgl.EID]]

            else:

                g.ndata['h'] = nfeats
                g.edata['h'] = efeats

            if self.aggregation == "mean":
                g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            if self.aggregation == "sum":
                g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            elif self.aggregation == "pool":
                g.update_all(self.message_func, fn.max('m', 'h_pool'))
                g.ndata['h_neigh'] = self.activation(
                    self.pool_fc(g.ndata['h_pool']))
            h_new = self.activation(self.W_apply(
                torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return h_new


class SAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, aggregation, dropout, num_neighbors):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    SAGELayer(ndim_in, edim, ndim_out[layer], activation, aggregation, num_neighbors[layer] if num_neighbors else None))
            else:
                self.layers.append(SAGELayer(
                    ndim_out[layer-1], edim, ndim_out[layer], activation, aggregation, num_neighbors[layer] if num_neighbors else None))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):

        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class EGRAPHSAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=True, num_class=2, num_neighbors=None, aggregation="mean"):
        super().__init__()
        self.gnn = SAGE(ndim_in, edim, ndim_out, num_layers,
                        activation, aggregation, dropout, num_neighbors)
        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

#############################
#############################
#############################
# E_GAT Model


class GATLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, num_neighbors=None):
        super(GATLayer, self).__init__()
        self.W_apply = nn.Linear(ndim_in + edim, ndim_out)
        self.attn_fc = nn.Linear(2*ndim_in, 1)
        self.activation = activation
        self.num_neighbors = num_neighbors

    def edge_attention(self, edges):
        return {"e": self.activation(self.attn_fc(torch.cat([edges.src["h"], edges.dst["h"]], dim=2)))}

    def message_func(self, edges):
        return {"m": edges.data['h'], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'z': z}

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            if self.num_neighbors:
                g = dgl.sampling.sample_neighbors(
                    g, g.nodes(), self.num_neighbors)
                g.ndata['h'] = nfeats
                g.edata['h'] = efeats[g.edata[dgl.EID]]
            else:
                g.ndata['h'] = nfeats
                g.edata['h'] = efeats
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            g.ndata['h'] = self.activation(self.W_apply(
                torch.cat([g.ndata['h'], g.ndata['z']], 2)))
            return g.ndata['h']


class GAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout, num_neighbors):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    GATLayer(ndim_in, edim, ndim_out[layer], activation, num_neighbors[layer] if num_neighbors else None))
            else:
                self.layers.append(
                    GATLayer(ndim_out[layer-1], edim, ndim_out[layer], activation, num_neighbors[layer] if num_neighbors else None))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class EGAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=False, num_class=2, num_neighbors=None):
        super().__init__()
        self.gnn = GAT(ndim_in, edim, ndim_out, num_layers,
                       activation, dropout, num_neighbors)

        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)


class Model:
    def __init__(
        self,
        model_name,
        model_class,
        num_layers,
        ndim_out,
        activation="relu",
        dropout=0.2,
        residual=False,
        aggregation=None,
        num_neighbors=None,
        norm=False,
        models=None,
    ):
        if models is None:
            models = {}
        self.model_name = model_name
        self.model_class = model_class
        self.num_layers = num_layers
        self.ndim_out = ndim_out
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.aggregation = aggregation
        self.num_neighbors = num_neighbors
        self.norm = norm
        self.models = models


# %% [markdown]
# ## Data Loading & Preprocessing
# 
# Load, clean and prepare the dataset. Includes optional graph metric analysis.

# %%
# Input validation
import os
if not os.path.exists(original_path):
    raise FileNotFoundError(f'Dataset file not found at {original_path}')
os.makedirs(folder_path, exist_ok=True)


# %%
# Preparing Datasets
datesetInfo = datasets[dataset_name]
df = pd.read_parquet(original_path)

timestamp_format = "mixed"
df = clean_dataset(df, flow_id_col=datesetInfo.flow_id_col,
                   timestamp_col=datesetInfo.timestamp_col)

df[datesetInfo.src_ip_col] = df[datesetInfo.src_ip_col].apply(str)
if datesetInfo.src_port_col:
    df[datesetInfo.src_port_col] = df[datesetInfo.src_port_col].apply(str)

df[datesetInfo.dst_ip_col] = df[datesetInfo.dst_ip_col].apply(str)
if datesetInfo.dst_port_col:
    df[datesetInfo.dst_port_col] = df[datesetInfo.dst_port_col].apply(str)

_, var_dropped, corr_dropped = feature_analysis_pipeline(
    df=df, drop_columns=datesetInfo.drop_columns, label_col=datesetInfo.label_col)
var_dropped, corr_dropped

var_dropped = set(var_dropped)
weak_columns = var_dropped.union(set(corr_dropped))

if with_sort_timestamp and datesetInfo.timestamp_col:
    df[datesetInfo.timestamp_col] = pd.to_datetime(
        df[datesetInfo.timestamp_col].str.strip(), format=timestamp_format)
    df.sort_values(datesetInfo.timestamp_col, inplace=True)

df, labels_names, classes = one_dataset_class_num_col(
    df, datesetInfo.class_num_col, datesetInfo.class_col)

os.makedirs(folder_path, exist_ok=True)
with open(folder_path + '/labels_names.pkl', 'wb') as f:
    pickle.dump([labels_names, classes], f)

# %%

# Optional: analyse dataset properties to inform adaptive GNN selection
if ANALYSE_GRAPH_METRICS:
    import numpy as np
    # Proportion of attackers (non‑zero labels) in the cleaned DataFrame
    attackers_ratio = (df[datesetInfo.label_col] != 0).mean()
    # Degree distribution based on source/destination IP counts
    from collections import Counter
    degrees = Counter()
    for col in [datesetInfo.src_ip_col, datesetInfo.dst_ip_col]:
        counts = df[col].value_counts()
        for node, cnt in counts.items():
            degrees[node] += cnt
    total_deg = sum(degrees.values())
    probs = np.array([deg/total_deg for deg in degrees.values()])
    # Entropy of degree distribution
    deg_entropy = -(probs * np.log(probs + 1e-12)).sum()
    print('Proportion of attackers:', attackers_ratio)
    print('Degree distribution entropy:', deg_entropy)
    # Suggest simple thresholds (can be tuned)
    print('Suggested high attacker threshold: 0.5')
    print('Suggested high entropy threshold: 1.0')


# %% [markdown]
# ## Graph Preparation
# 
# Construct graphs from the preprocessed dataset.

# %%
# prepare_graph_files.ipynb
cols_to_norm = list(set(list(df.columns))  - set(list([datesetInfo.label_col, datesetInfo.class_num_col])) - set(datesetInfo.drop_columns)  - set(datesetInfo.weak_columns))

if generated_ips:
    df[datesetInfo.src_ip_col] = df[datesetInfo.src_ip_col].apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))



if sort_timestamp:
    df[datesetInfo.timestamp_col] = pd.to_datetime(df[datesetInfo.timestamp_col].str.strip(), format=datesetInfo.timestamp_format)
    df.sort_values(datesetInfo.timestamp_col, inplace=True)

if use_port_in_address:
    df[datesetInfo.src_port_col] = df[datesetInfo.src_port_col].astype(float).astype(int).astype(str) # to remove the decimal point
    df[datesetInfo.src_ip_col] = df[datesetInfo.src_ip_col] + ':' + df[datesetInfo.src_port_col]

    df[datesetInfo.dst_port_col] = df[datesetInfo.dst_port_col].astype(float).astype(int).astype(str) # to remove the decimal point
    df[datesetInfo.dst_ip_col] = df[datesetInfo.dst_ip_col] + ':' + df[datesetInfo.dst_port_col]

if multi_class:
    y = df[datesetInfo.class_num_col]
else:
    y = df[datesetInfo.label_col]

if sort_timestamp:
    X_tr, X_test, y_tr, y_test = train_test_split(
        df, y, test_size=test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=validation_size)
else:
    X_tr, X_test, y_tr, y_test = train_test_split(
        df, y, test_size=test_size, random_state=13, stratify=y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=validation_size, random_state=13, stratify=y_tr)

# del df

if graph_type == "line" and use_node_features:
    add_centralities(df = X_train, new_path=None, graph_path=None, dataset=datesetInfo, cn_measures=cn_measures, network_features=network_features, create_using=nx.MultiDiGraph())
    add_centralities(df = X_val, new_path=None, graph_path=None, dataset=datesetInfo, cn_measures=cn_measures, network_features=network_features, create_using=nx.MultiDiGraph())
    add_centralities(df = X_test, new_path=None, graph_path=None, dataset=datesetInfo, cn_measures=cn_measures, network_features=network_features, create_using=nx.MultiDiGraph())
    cols_to_norm = list(set(cols_to_norm) | set(network_features))


scaler = StandardScaler()

X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_train['h'] = X_train[ cols_to_norm ].values.tolist()

cols_to_drop = list(set(list(X_train.columns)) - set(list([datesetInfo.label_col, datesetInfo.src_ip_col, datesetInfo.dst_ip_col, datesetInfo.class_num_col, 'h'])))
X_train.drop(cols_to_drop, axis=1, inplace=True)

X_val[cols_to_norm] = scaler.transform(X_val[cols_to_norm])
X_val['h'] = X_val[ cols_to_norm ].values.tolist()
X_val.drop(cols_to_drop, axis=1, inplace=True)

X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[ cols_to_norm ].values.tolist()
X_test.drop(cols_to_drop, axis=1, inplace=True)

if graph_type == "window" or graph_type == "line":

    create_weightless_window_graph(
        df=X_train,
        dataset=datesetInfo,
        window_size=window_size,
        line_graph=graph_type == "line",
        folder_path=os.path.join(folder_path, "training"),
        edge_attr= ['h', datesetInfo.label_col, datesetInfo.class_num_col],
        file_type="pkl")
    
    create_weightless_window_graph(
        df=X_val,
        dataset=datesetInfo,
        window_size=window_size,
        line_graph=graph_type == "line",
        folder_path=os.path.join(folder_path, "validation"),
        edge_attr= ['h', datesetInfo.label_col, datesetInfo.class_num_col],
        file_type="pkl")
    
    create_weightless_window_graph(
        df=X_test,
        dataset=datesetInfo,
        window_size=window_size,
        line_graph=graph_type == "line",
        folder_path=os.path.join(folder_path, "testing"),
        edge_attr= ['h', datesetInfo.label_col, datesetInfo.class_num_col],
        file_type="pkl")

if graph_type == "flow":
	os.makedirs(folder_path, exist_ok=True)
	print(f"==>> X_train.shape: {X_train.shape}")
	print(f"==>> X_val.shape: {X_val.shape}")
	print(f"==>> X_test.shape: {X_test.shape}")
if graph_type == "flow":
    graph_name = "training_graph"

    G = nx.from_pandas_edgelist(X_train, datesetInfo.src_ip_col, datesetInfo.dst_ip_col, ['h',datesetInfo.label_col, datesetInfo.class_num_col], create_using=nx.MultiDiGraph())
    
    if use_node_features:
        add_centralities_as_node_features(df=None, G=G, graph_path=None, dataset=datesetInfo, cn_measures=cn_measures)
        
        for node in G.nodes():
            centralities = []
            for centrality in cn_measures:
                centralities.append(G.nodes[node].get(centrality, 0)) # Default to 0 if missing
                
                # Combine features into a single vector
            n_feats = np.array(centralities, dtype=np.float32)
            
            # Add the new feature to the node
            G.nodes[node]["n_feats"] = n_feats
            
    # get netowrk properties
    graph_measures = calculate_graph_measures(G, f"{folder_path}/{graph_name}_measures.json", verbose=True)
    print(f"==>> graph_measures: {graph_measures}")

    # graph_measures = calculate_graph_measures(nx.DiGraph(G), "datasets/" + name + "/training_graph_simple_measures.json", verbose=True)
    # print(f"==>> graph_measures: {graph_measures}")

    with open(f"{folder_path}/{graph_name}.pkl", "wb") as f:
        pickle.dump(G, f)

if graph_type == "flow":
    graph_name = "validation_graph"

    G = nx.from_pandas_edgelist(X_val, datesetInfo.src_ip_col, datesetInfo.dst_ip_col, ['h',datesetInfo.label_col, datesetInfo.class_num_col], create_using=nx.MultiDiGraph())
    
    if use_node_features:
        add_centralities_as_node_features(df=None, G=G, graph_path=None, dataset=datesetInfo, cn_measures=cn_measures)
        
        for node in G.nodes():
            centralities = []
            for centrality in cn_measures:
                centralities.append(G.nodes[node].get(centrality, 0)) # Default to 0 if missing
                
                # Combine features into a single vector
            n_feats = np.array(centralities, dtype=np.float32)
            
            # Add the new feature to the node
            G.nodes[node]["n_feats"] = n_feats
            
    # get netowrk properties
    graph_measures = calculate_graph_measures(G, f"{folder_path}/{graph_name}_measures.json", verbose=True)
    print(f"==>> graph_measures: {graph_measures}")

    # graph_measures = calculate_graph_measures(nx.DiGraph(G), "datasets/" + name + "/training_graph_simple_measures.json", verbose=True)
    # print(f"==>> graph_measures: {graph_measures}")

    with open(f"{folder_path}/{graph_name}.pkl", "wb") as f:
        pickle.dump(G, f)

if graph_type == "flow":
    graph_name = "testing_graph"
    
    G = nx.from_pandas_edgelist(X_test, datesetInfo.src_ip_col, datesetInfo.dst_ip_col, ['h', datesetInfo.label_col, datesetInfo.class_num_col], create_using=nx.MultiDiGraph())
    
    if use_node_features:
        add_centralities_as_node_features(df=None, G=G, graph_path=None, dataset=datesetInfo, cn_measures=cn_measures)
        
        for node in G.nodes():
            centralities = []
            for centrality in cn_measures:
                centralities.append(G.nodes[node].get(centrality, 0)) # Default to 0 if missing
                
                # Combine features into a single vector
            n_feats = np.array(centralities, dtype=np.float32)
            
            # Add the new feature to the node
            G.nodes[node]["n_feats"] = n_feats
            
    graph_measures = calculate_graph_measures(G, f"{folder_path}/{graph_name}_measures.json", verbose=True)
    print(f"==>> graph_measures: {graph_measures}")
    
    # graph_measures = calculate_graph_measures(nx.DiGraph(G_test), "datasets/" + name + "/testing_graph_simple_measures.json", verbose=True)
    # print(f"==>> graph_measures: {graph_measures}")
    
    with open(f"{folder_path}/{graph_name}.pkl", "wb") as f:
        pickle.dump(G, f)

# %% [markdown]
# ### Speed/Memory Tips
# 
# Use flags to trade off speed and memory.
# - `USE_MIXED_PRECISION`: enables 16‑bit mixed precision training.
# - `USE_GRADIENT_ACCUMULATION`: accumulates gradients over two batches.
# - `USE_ENHANCED_LOGGING`: adds extra callbacks for richer logging.
# 

# %% [markdown]
# ## Training & Evaluation
# 
# Train and evaluate models. Optional reviewer‑requested features are controlled via flags.

# %%
import warnings, time
warnings.filterwarnings('ignore', '.*does not have many workers.*')

def run_training_for_seed(run_seed):
    # Optionally reseed deterministically for each run
    if USE_MULTI_SEED_EVAL or USE_DETERMINISTIC:
        seed_everything_deterministic(run_seed)
    datesetInfo = datasets[dataset_name]
    activation = F.relu
    graphs_folder_local = folder_path if graph_type == 'flow' else os.path.join(folder_path, g_type)
    # Ensure log dirs
    logs_folder = os.path.join('logs', datesetInfo.name)
    os.makedirs(logs_folder, exist_ok=True)
    wandb_runs_path = os.path.join('logs', 'wandb_runs')
    os.makedirs(wandb_runs_path, exist_ok=True)
    # Label mapping
    labels_mapping = {0: 'Normal', 1: 'Attack'}
    if multi_class:
        with open(os.path.join(folder_path, 'labels_names.pkl'), 'rb') as f:
            labels_names = pickle.load(f)
        labels_mapping = labels_names[0]
    num_classes_local = len(labels_mapping)
    # Dataset kwargs
    dataset_kwargs = dict(
        use_node_features=use_centralities_nfeats,
        multi_class=True,
        using_masking=False,
        masked_class=2,
        num_workers=0,
        label_col=datesetInfo.label_col,
        class_num_col=datesetInfo.class_num_col,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Data module
    data_module = GraphDataModule(graphs_folder_local, batch_size=1, **dataset_kwargs)
    data_module.setup()
    ndim = next(iter(data_module.train_dataloader())).ndata['h'].shape[-1]
    edim = next(iter(data_module.train_dataloader())).edata['h'].shape[-1]
    # Build models dictionary
    models_dict = {}
    #     'e_gat_no_sampling': EGAT(ndim, edim, ndim_out, num_layers, activation, dropout,
    #       residual, len(labels_mapping), num_neighbors=None)
    # }
    # Include GCN and GraphSAGE variants with and without sampling
    # models_dict['e_gcn'] = EGCN(ndim, edim, ndim_out, num_layers, activation, dropout, residual, len(labels_mapping))
    # models_dict[f'e_graphsage_{aggregation}'] = EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
    #       residual, len(labels_mapping), num_neighbors=number_neighbors, aggregation=aggregation)
    # models_dict[f'e_graphsage_{aggregation}_no_sampling'] = EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
    #       residual, len(labels_mapping), num_neighbors=None, aggregation=aggregation)
    # Loss criterion

    C = len(labels_mapping)
    try:
        # if your DataModule exposes counts
        counts = torch.as_tensor(data_module.train_dataset.class_counts)
    except Exception:
        # fallback: compute from the train dataloader once
        counts = torch.zeros(C, dtype=torch.long)
        for g in data_module.train_dataloader():
            # try common label keys
            if datesetInfo.label_col in g.edata:
                yb = g.edata[datesetInfo.label_col]
            else:
                raise KeyError("Could not find labels in graph batch (expected edata['label'] or edata['y']).")
            counts += torch.bincount(yb.to(torch.long), minlength=C)
        counts = counts.clamp_min(1)

    # Handle focal_alpha configuration
    if focal_alpha == "weighted_class_counts":
        # Use alpha_from_counts to compute weighted alpha tensor
        alpha_tensor = alpha_from_counts(
            counts=counts,
            scheme=class_counts_scheme,
            beta=class_counts_beta,
            normalize=class_counts_normalize
        )
        focal_alpha_processed = alpha_tensor
    else:
        focal_alpha_processed = focal_alpha if (focal_alpha is None or isinstance(focal_alpha,(float,int))) else torch.tensor(focal_alpha, dtype=torch.float32)

    criterion, needs_epoch_hook = build_imbalance_loss(
        loss_name=loss_name,
        num_classes=C,
        counts=counts,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha_processed,
        cb_beta=cb_beta,
        ldam_C_margin=ldam_C_margin,
        drw_start=drw_start,
        cb_beta_drw=cb_beta_drw,
        logit_adj_tau=logit_adj_tau,
    )


    # if use_focal_loss:
    #     criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    # else:
    #     criterion = nn.CrossEntropyLoss(data_module.train_dataset.class_weights)
    seed_results = {}
    for model_name, model in models_dict.items():
        config = {
            'run_dtime': run_dtime,
            'type': 'GNN',
            'model_name': model_name,
            'max_epochs': max_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'ndim_out': ndim_out,
            'num_layers': num_layers,
            'number_neighbors': number_neighbors,
            'activation': activation.__name__,
            'dropout': dropout,
            'residual': residual,
            'multi_class': multi_class,
            'aggregation': aggregation,
            'loss_name': loss_name,
            'class_counts_scheme': class_counts_scheme,
            'class_counts_beta': class_counts_beta,
            'class_counts_normalize': class_counts_normalize,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'cb_beta': cb_beta,
            'ldam_C_margin': ldam_C_margin,
            'drw_start': drw_start,
            'cb_beta_drw': cb_beta_drw,
            'logit_adj_tau': logit_adj_tau,
            'early_stopping_patience': early_stopping_patience,
            'use_centralities_nfeats': use_centralities_nfeats,
            'USE_ENHANCED_LOGGING': USE_ENHANCED_LOGGING,
            'use_centralities_nfeats': use_centralities_nfeats,
            'USE_DETERMINISTIC': USE_DETERMINISTIC,
            'USE_MIXED_PRECISION': USE_MIXED_PRECISION,
            'USE_GRADIENT_ACCUMULATION': USE_GRADIENT_ACCUMULATION,
            'USE_EXTRA_METRICS': USE_EXTRA_METRICS,
            'USE_MULTI_SEED_EVAL': USE_MULTI_SEED_EVAL,
            'USE_COMPLEXITY_LOGGING': USE_COMPLEXITY_LOGGING,
            'ANALYSE_GRAPH_METRICS': ANALYSE_GRAPH_METRICS,
        }
        graph_model = GraphModel(model, criterion, learning_rate, config, model_name,
                                 labels_mapping, weight_decay=weight_decay, using_wandb=using_wandb, norm=False, multi_class=True)
        # W&B logger
        if using_wandb:
            wandb_logger = WandbLogger(
                project=f'GNN-Analysis-{datesetInfo.name}',
                name=model_name,
                config=config,
                save_dir=wandb_runs_path
            )
        else:
            wandb_logger = None
        # Callbacks
        f1_checkpoint_callback = ModelCheckpoint(monitor='val_f1_score', mode='max', filename='best-val-f1-{epoch:02d}-{val_f1_score:.2f}', save_top_k=save_top_k, verbose=False)
        early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience, verbose=False)
        extra_callbacks = []
        if USE_ENHANCED_LOGGING:
            extra_callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
            extra_callbacks.append(pl.callbacks.ModelSummary(max_depth=2))
        trainer_kwargs = {}
        if USE_MIXED_PRECISION:
            trainer_kwargs['precision'] = '16-mixed'
        if USE_GRADIENT_ACCUMULATION:
            trainer_kwargs['accumulate_grad_batches'] = 2
        if USE_COMPLEXITY_LOGGING:
            start_time = time.time()
        trainer = pl.Trainer(max_epochs=max_epochs, num_sanity_val_steps=0, log_every_n_steps=0, callbacks=[f1_checkpoint_callback, early_stopping_callback] + extra_callbacks, default_root_dir=logs_folder, logger=wandb_logger if using_wandb else None, **trainer_kwargs)
        trainer.fit(graph_model, datamodule=data_module)
        # Test each best checkpoint
        test_scores = []
        for i, k in enumerate(f1_checkpoint_callback.best_k_models.keys()):
            graph_model.test_prefix = f'best_f1_{i}'
            results = trainer.test(graph_model, datamodule=data_module, ckpt_path=k)
            test_scores.append(results[0][f'best_f1_{i}_test_f1'])
        logs = {
            'median_f1_of_best_f1': np.median(test_scores),
            'max_f1_of_best_f1': np.max(test_scores),
            'avg_f1_of_best_f1': np.mean(test_scores)
        }
        # Additional metrics if enabled
        if USE_EXTRA_METRICS:
            # Evaluate on full test set using current best checkpoint
            # Note: for simplicity we reuse the same test_scores for F1 and derive precision/recall from confusion matrix
            y_true = []
            y_pred = []
            # Use data loader to iterate through test dataset
            test_loader = data_module.test_dataloader()
            graph_model.eval()
            for batch in test_loader:
                g, labels = batch
                preds = graph_model(g, g.ndata['h']).argmax(dim=1).detach().cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy())
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            logs['accuracy'] = accuracy_score(y_true, y_pred)
            logs['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            logs['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        if USE_COMPLEXITY_LOGGING:
            end_time = time.time()
            train_time = end_time - start_time
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logs['train_time_seconds'] = train_time
            logs['num_parameters'] = num_params
        # Log results
        if using_wandb:
            wandb.log(logs)
            wandb.finish()
        else:
            if trainer.logger:
                trainer.logger.log_metrics(logs, step=trainer.global_step)
        # Save metrics for multi‑seed aggregation
        seed_results[model_name] = logs['avg_f1_of_best_f1']
    return seed_results

# Multi‑seed evaluation
if USE_MULTI_SEED_EVAL:
    seeds = [CONFIG.seed + i for i in range(5)]
    all_results = {}
    for s in seeds:
        res = run_training_for_seed(s)
        for m, score in res.items():
            all_results.setdefault(m, []).append(score)
    # Report mean and std for each model
    for m, scores in all_results.items():
        mean_f1 = np.mean(scores)
        std_f1 = np.std(scores)
        print(f'{m}: mean F1 = {mean_f1:.4f} ± {std_f1:.4f}')
        if using_wandb:
            wandb.log({f'{m}_mean_f1': mean_f1, f'{m}_std_f1': std_f1})
else:
    # Single‑seed run
    run_training_for_seed(CONFIG.seed)


# %%
try:
    print('Trainer configuration summary:')
    print({'max_epochs': trainer.max_epochs,
           'callbacks': [type(cb).__name__ for cb in trainer.callbacks],
           'logger': type(trainer.logger).__name__ if hasattr(trainer, 'logger') else None})
except Exception as e:
    print('Trainer not defined or training not yet run:', e)

# %% [markdown]
# ## Troubleshooting
# 
# Common issues and remedies:
# 
# - **Out of memory errors**: enable `USE_MIXED_PRECISION` or `USE_GRADIENT_ACCUMULATION`.
# - **Missing data**: ensure `original_path` points to a valid Parquet file.
# - **Slow training**: reduce neighbours or layers, or use mixed precision.
# - **Authentication issues**: configure W&B API key appropriately.
# - **Reproducibility concerns**: enable `USE_DETERMINISTIC` or use multi‑seed evaluation.
# - **Statistical significance**: enable `USE_MULTI_SEED_EVAL` to compute mean±std across seeds.
# 


