import torch.nn as nn
import os
import pickle
import torch
import warnings
import wandb
import time
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from src.models import EGAT, EGCN, EGRAPHSAGE
from src.lightning_model import GraphModel
from src.lightning_data import GraphDataModule
from src.dataset.dataset_info import datasets
from local_variables import local_datasets_path

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def main():
    using_wandb = True
    save_top_k = 5

    # Hyperparameters
    # dataset_name = "cic_ton_iot_5_percent"
    # dataset_name = "cic_ton_iot"
    dataset_name = "cic_ids_2017_5_percent"
    # dataset_name = "cic_ids_2017"
    # dataset_name = "cic_bot_iot"
    # dataset_name = "cic_ton_iot_modified"
    # dataset_name = "nf_ton_iotv2_modified"
    # dataset_name = "ccd_inid_modified"
    # dataset_name = "nf_uq_nids_modified"
    # dataset_name = "edge_iiot"
    # dataset_name = "nf_cse_cic_ids2018"
    # dataset_name = "nf_bot_iotv2"
    # dataset_name = "nf_uq_nids"
    # dataset_name = "x_iiot"

    max_epochs = 30
    early_stopping_patience = 20
    learning_rate = 0.001
    weight_decay = 0.001
    ndim_out = [128]
    num_layers = 1
    number_neighbors = [25]
    activation = F.relu
    dropout = 0.2
    residual = True
    multi_class = True
    aggregation = "mean"

    run_dtime = time.strftime("%Y%m%d-%H%M%S")

    dataset = datasets[dataset_name]
    dataset_folder = os.path.join(local_datasets_path, dataset.name)
    graphs_folder = os.path.join(dataset_folder, "flow__multi_class__unsorted")
    logs_folder = os.path.join("logs", dataset.name)
    os.makedirs(logs_folder, exist_ok=True)
    wandb_runs_path = os.path.join("logs", "wandb_runs")
    os.makedirs(wandb_runs_path, exist_ok=True)

    labels_mapping = {0: "Normal", 1: "Attack"}
    num_classes = 2
    if multi_class:
        with open(os.path.join(dataset_folder, "labels_names.pkl"), "rb") as f:
            labels_names = pickle.load(f)
        labels_mapping = labels_names[0]
    num_classes = len(labels_mapping)

    dataset_kwargs = dict(
        use_node_features=False,
        multi_class=True,
        using_masking=False,
        masked_class=2,
        num_workers=0,
        label_col=dataset.label_col,
        class_num_col=dataset.class_num_col,
        device='cuda' if torch.cuda.is_available() else "cpu"
    )

    data_module = GraphDataModule(
        graphs_folder, batch_size=1, **dataset_kwargs)
    data_module.setup()

    ndim = next(iter(data_module.train_dataloader())).ndata["h"].shape[-1]
    edim = next(iter(data_module.train_dataloader())).edata['h'].shape[-1]

    my_models = {
        "e_gat_sampling": EGAT(ndim, edim, ndim_out, num_layers, activation, dropout,
                               residual, num_classes, num_neighbors=number_neighbors),
        "e_gcn": EGCN(ndim, edim, ndim_out, num_layers, activation,
                      dropout, residual, num_classes),
        f"e_graphsage_{aggregation}": EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
                                                 residual, num_classes, num_neighbors=number_neighbors, aggregation=aggregation),
        # f"e_graphsage_{aggregation}_no_sampling": EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
        #                                                      residual, num_classes, num_neighbors=None, aggregation=aggregation),

        "e_gat_no_sampling": EGAT(ndim, edim, ndim_out, num_layers, activation, dropout,
                                  residual, num_classes, num_neighbors=None),
    }

    criterion = nn.CrossEntropyLoss(data_module.train_dataset.class_weights)

    for model_name, model in my_models.items():

        config = {
            "run_dtime": run_dtime,
            "model_name": model_name,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "ndim_out": ndim_out,
            "num_layers": num_layers,
            "number_neighbors": number_neighbors,
            "activation": activation.__name__,
            "dropout": dropout,
            "residual": residual,
            "multi_class": multi_class,
            "aggregation": aggregation,
            "early_stopping_patience": early_stopping_patience,
        }

        graph_model = GraphModel(model, criterion, learning_rate, config, model_name,
                                 labels_mapping, weight_decay=weight_decay, using_wandb=using_wandb, norm=False, multi_class=True)

        if using_wandb:
            wandb_logger = WandbLogger(
                project=f"GNN-Analysis-{dataset.name}",
                name=model_name,
                config=config,
                save_dir=wandb_runs_path
            )
        else:
            wandb_logger = None

        f1_checkpoint_callback = ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            filename="best-val-f1-{epoch:02d}-{val_f1_score:.2f}",
            save_top_k=save_top_k,
            save_on_train_epoch_end=False,
            verbose=False,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            verbose=False,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            # log_every_n_steps=0,
            callbacks=[
                f1_checkpoint_callback,
                early_stopping_callback
            ],
            default_root_dir=logs_folder,
            logger=wandb_logger,
        )

        trainer.fit(graph_model, datamodule=data_module)

        test_results = []
        print(
            f"==>> f1_checkpoint_callback.best_k_models.keys(): {f1_checkpoint_callback.best_k_models.keys()}")
        for i, k in enumerate(f1_checkpoint_callback.best_k_models.keys()):
            graph_model.test_prefix = f"best_f1_{i}"
            results = trainer.test(
                graph_model, datamodule=data_module, ckpt_path=k)
            test_results.append(results[0][f"best_f1_{i}_test_f1"])

        logs = {
            "median_f1_of_best_f1": np.median(test_results),
            "max_f1_of_best_f1": np.max(test_results),
            "avg_f1_of_best_f1": np.mean(test_results)
        }

        if using_wandb:
            wandb.log(logs)
            wandb.finish()
        else:
            trainer.logger.log_metrics(logs, step=trainer.global_step)


if __name__ == "__main__":
    main()
