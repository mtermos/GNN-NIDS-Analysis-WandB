import torch.nn as nn
import os
import pickle
import torch
import warnings
import wandb
import time
import random
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from src.models import EGCN, EGRAPHSAGE, EGAT, EGIN
# from src.models import EGAT, EGCN, EGRAPHSAGE
from src.lightning_model import GraphModel, WindowedGraphModel
from src.lightning_data import GraphDataModule
from src.dataset.dataset_info import datasets
from local_variables import local_datasets_path
# Import the new config
from src.config import CONFIG
from src.graph.create_graph_files import create_graph_files
from src.loss_functions import alpha_from_counts, build_imbalance_loss

warnings.filterwarnings("ignore", ".*does not have many workers.*")

os.environ["DGLBACKEND"] = "pytorch"

def main():
    # Use the config from the separate file
    config = CONFIG
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    try:
        torch.cuda.manual_seed_all(config.seed)
    except Exception:
        pass
    pl.seed_everything(config.seed, workers=True)

    # You can also customize specific parameters if needed:
    # config = get_config(dataset_name="cic_ton_iot_5_percent", max_epochs=10)
    
    # Print configuration summary
    config.print_config_summary()
    
    dataset = datasets[config.dataset_name]
    dataset_folder = os.path.join(local_datasets_path, dataset.name)
    graphs_folder = os.path.join(dataset_folder, config.g_type)


    ### if graphs_folder doesn't exist, then create it
    if not os.path.exists(graphs_folder):
        create_graph_files(config.original_path, dataset, config, graphs_folder)


    logs_folder = os.path.join("logs", dataset.name)
    os.makedirs(logs_folder, exist_ok=True)
    wandb_runs_path = os.path.join("logs", "wandb_runs")
    os.makedirs(wandb_runs_path, exist_ok=True)

    labels_mapping = {0: "Normal", 1: "Attack"}
    num_classes = 2
    if config.multi_class:
        with open(os.path.join(graphs_folder, "labels_names.pkl"), "rb") as f:
            labels_names = pickle.load(f)
        labels_mapping = labels_names[0]
    num_classes = len(labels_mapping)

    # Handle device selection for Apple Silicon compatibility
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Force CPU on Apple Silicon since DGL doesn't support MPS
        device = 'cpu'
        print("MPS detected but not supported by DGL. Using CPU instead.")
    else:
        device = 'cpu'
    
    dataset_kwargs = dict(
        use_node_features=config.use_centralities_nfeats,
        multi_class=True,
        using_masking=False,
        masked_class=2,
        num_workers=0,
        label_col=dataset.label_col,
        class_num_col=dataset.class_num_col,
        device=device
    )
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_module = GraphDataModule(
        graphs_folder, config.graph_type, batch_size=config.batch_size, **dataset_kwargs)
    data_module.setup()

    ndim = next(iter(data_module.train_dataloader())).ndata["h"].shape[-1]
    edim = next(iter(data_module.train_dataloader())).edata['h'].shape[-1]

    # Support string names for common activations, or pass a callable directly
    if isinstance(config.activation, str):
        if config.activation.lower() == "relu":
            activation = F.relu
        elif config.activation.lower() == "gelu":
            activation = F.gelu
        elif config.activation.lower() == "tanh":
            activation = torch.tanh
        elif config.activation.lower() == "sigmoid":
            activation = torch.sigmoid
        elif config.activation.lower() == "leaky_relu":
            activation = F.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
    else:
        activation = config.activation
    
    all_models = {
        "e_graphsage": EGRAPHSAGE(
            ndim, edim, config.ndim_out, config.num_layers, activation, config.dropout,
            config.residual, num_classes, num_neighbors=config.number_neighbors, aggregation=config.aggregation, edge_update=config.edge_update
        ),
        "e_gat": EGAT(
            ndim, edim, config.ndim_out, config.num_layers, activation, config.dropout,
            config.residual, num_classes, num_neighbors=config.number_neighbors, edge_update=config.edge_update
        ),
    }

    my_models = {k: all_models[k] for k in config.selected_models}

    C = len(labels_mapping)
    try:
        # if your DataModule exposes counts
        counts = torch.as_tensor(data_module.train_dataset.class_counts)
    except Exception:
        # fallback: compute from the train dataloader once
        counts = torch.zeros(C, dtype=torch.long)
        for g in data_module.train_dataloader():
            # try common label keys
            if dataset.label_col in g.edata:
                yb = g.edata[dataset.label_col]
            else:
                raise KeyError("Could not find labels in graph batch (expected edata['label'] or edata['y']).")
            counts += torch.bincount(yb.to(torch.long), minlength=C)
        counts = counts.clamp_min(1)

    # Handle focal_alpha configuration
    if config.focal_alpha == "weighted_class_counts":
        # Use alpha_from_counts to compute weighted alpha tensor
        alpha_tensor = alpha_from_counts(
            counts=counts,
            scheme=config.class_counts_scheme,
            beta=config.class_counts_beta,
            normalize=config.class_counts_normalize
        )
        focal_alpha_processed = alpha_tensor
    else:
        focal_alpha_processed = config.focal_alpha if (config.focal_alpha is None or isinstance(config.focal_alpha,(float,int))) else torch.tensor(config.focal_alpha, dtype=torch.float32)

    criterion, needs_epoch_hook = build_imbalance_loss(
        loss_name=config.loss_name,
        num_classes=C,
        counts=counts,
        focal_gamma=config.focal_gamma,
        focal_alpha=focal_alpha_processed,
        cb_beta=config.cb_beta,
        ldam_C_margin=config.ldam_C_margin,
        drw_start=config.drw_start,
        cb_beta_drw=config.cb_beta_drw,
        logit_adj_tau=config.logit_adj_tau,
    )

    # criterion = nn.CrossEntropyLoss(data_module.train_dataset.class_weights)

    elapsed = {
        "dataset": dataset.name
    }

    for model_name, model in my_models.items():

        # Add experiment-unique fields to a copy of the original config
        config_dict = config.get_config_summary() if hasattr(config, "get_config_summary") else dict(config)
        config_dict = dict(config_dict)  # make a copy to avoid mutating the original

        config_dict.update({
            "type": "GNN",
            "model_name": model_name,
            "ndim": ndim,
            "edim": edim,
        })

        if config.graph_type == "flow":
            graph_model = GraphModel(model, criterion, config.learning_rate, config_dict, model_name,
                                     labels_mapping, weight_decay=config.weight_decay, using_wandb=config.using_wandb, norm=False, multi_class=True,
                                     needs_epoch_hook=needs_epoch_hook)
        elif config.graph_type == "window":
            graph_model = WindowedGraphModel(model, criterion, config.learning_rate, config_dict, model_name,
                                             labels_mapping, weight_decay=config.weight_decay, using_wandb=config.using_wandb, norm=False, multi_class=True,
                                             needs_epoch_hook=needs_epoch_hook)

        if config.using_wandb:
            wandb_logger = WandbLogger(
                project=f"GNN-Analysis-{dataset.name}",
                name=model_name,
                config=config_dict,
                save_dir=wandb_runs_path
            )
        else:
            wandb_logger = None

        f1_checkpoint_callback = ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            filename="best-val-f1-{epoch:02d}-{val_f1_score:.2f}",
            save_top_k=config.save_top_k,
            save_on_train_epoch_end=False,
            verbose=False,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            verbose=False,
        )

        extra_callbacks = []
        if config.use_enhanced_logging:
            extra_callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='step'))
            extra_callbacks.append(pl.callbacks.ModelSummary(max_depth=2))

        trainer_kwargs = {}
        if config.use_mixed_precision:
            trainer_kwargs['precision'] = '16-mixed'
        if config.use_gradient_accumulation:
            trainer_kwargs['accumulate_grad_batches'] = 2
        if config.use_complexity_logging:
            start_time = time.time()

        # Force CPU on Apple Silicon to avoid DGL MPS compatibility issues
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            trainer_kwargs['accelerator'] = 'cpu'
            print("Forcing CPU accelerator for Apple Silicon compatibility with DGL")
        
        trainer = pl.Trainer(max_epochs=config.max_epochs, num_sanity_val_steps=0, log_every_n_steps=0, callbacks=[f1_checkpoint_callback, early_stopping_callback] + extra_callbacks, default_root_dir=logs_folder, logger=wandb_logger if config.using_wandb else None, **trainer_kwargs)
        trainer.fit(graph_model, datamodule=data_module)

        test_results = []
        test_elapsed = []
        print(
            f"==>> f1_checkpoint_callback.best_k_models.keys(): {f1_checkpoint_callback.best_k_models.keys()}")
        for i, k in enumerate(f1_checkpoint_callback.best_k_models.keys()):
            graph_model.test_prefix = f"best_f1_{i}"
            results = trainer.test(
                graph_model, datamodule=data_module, ckpt_path=k)
            test_results.append(results[0][f"best_f1_{i}_test_f1"])
            # test_elapsed.append(results[0][f"best_f1_{i}_elapsed"])

        logs = {
            "median_f1_of_best_f1": np.median(test_results),
            "max_f1_of_best_f1": np.max(test_results),
            "avg_f1_of_best_f1": np.mean(test_results)
        }
        # elapsed[model_name] = np.mean(test_elapsed).item()
        print(f"==>> model_name: {model_name}")
        # print(f"==>> test_elapsed: {np.mean(test_elapsed)}")

        if config.use_extra_metrics:
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
        if config.use_complexity_logging:
            end_time = time.time()
            train_time = end_time - start_time
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logs['train_time_seconds'] = train_time
            logs['num_parameters'] = num_params

        if config.using_wandb:
            wandb.log(logs)
            wandb.finish()
        else:
            if trainer.logger:
                trainer.logger.log_metrics(logs, step=trainer.global_step)
                
    print(f"==>> elapsed: {elapsed}")


if __name__ == "__main__":
    main()
