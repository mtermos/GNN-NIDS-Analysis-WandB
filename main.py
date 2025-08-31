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


from src.models import EGAT, EGRAPHSAGE, EGIN
# from src.models import EGAT, EGCN, EGRAPHSAGE
from src.lightning_model import GraphModel, WindowedGraphModel
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
    # dataset_name = "cic_ids_2017_5_percent"
    # dataset_name = "cic_ids_2017"
    # dataset_name = "cic_bot_iot"
    # dataset_name = "cic_ton_iot_modified"
    # dataset_name = "nf_ton_iotv2_modified"
    dataset_name = "ccd_inid_modified"
    # dataset_name = "nf_uq_nids_modified"
    # dataset_name = "edge_iiot"
    # dataset_name = "nf_cse_cic_ids2018"
    # dataset_name = "nf_bot_iotv2"
    # dataset_name = "nf_uq_nids"
    # dataset_name = "x_iiot"

    early_stopping_patience = max_epochs = 100
    # early_stopping_patience = 20
    learning_rate = 0.005
    weight_decay = 0.0
    ndim_out = [128, 128]
    num_layers = 2
    number_neighbors = [25, 10]
    activation = F.relu
    dropout = 0.0
    residual = True
    multi_class = True
    use_centralities_nfeats = False
    aggregation = "mean"

    sort_timestamp = False
    # sort_timestamp = True

    run_dtime = time.strftime("%Y%m%d-%H%M%S")

    graph_type = "flow"
    # graph_type = "window"
    batch_size = 1
    # graph_type = "line"

    window_size = 1000

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

    dataset = datasets[dataset_name]
    dataset_folder = os.path.join(local_datasets_path, dataset.name)
    graphs_folder = os.path.join(dataset_folder, g_type)

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
        use_node_features=use_centralities_nfeats,
        multi_class=True,
        using_masking=False,
        masked_class=2,
        num_workers=0,
        label_col=dataset.label_col,
        class_num_col=dataset.class_num_col,
        device='cuda' if torch.cuda.is_available() else "cpu"
    )

    data_module = GraphDataModule(
        graphs_folder, graph_type, batch_size=batch_size, **dataset_kwargs)
    data_module.setup()

    ndim = next(iter(data_module.train_dataloader())).ndata["h"].shape[-1]
    edim = next(iter(data_module.train_dataloader())).edata['h'].shape[-1]

    my_models = {
        # "e_gcn": EGCN(ndim, edim, ndim_out, num_layers, activation,
        #               dropout, residual, num_classes),
        # f"e_graphsage_{aggregation}": EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
        #                                          residual, num_classes, num_neighbors=number_neighbors, aggregation=aggregation),
        # f"e_graphsage_{aggregation}_no_sampling": EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
        #                                                      residual, num_classes, num_neighbors=None, aggregation=aggregation),

        # "e_gat_no_sampling": EGAT(ndim, edim, ndim_out, num_layers, activation, dropout,
        #                           residual, num_classes, num_neighbors=None),
        # "e_gat_sampling": EGAT(ndim, edim, ndim_out, num_layers, activation, dropout,
        #                        residual, num_classes, num_neighbors=number_neighbors),
    }

    criterion = nn.CrossEntropyLoss(data_module.train_dataset.class_weights)

    elapsed = {
        "dataset": dataset.name
    }
    for model_name, model in my_models.items():

        config = {
            "run_dtime": run_dtime,
            "type": "GNN",
            "model_name": model_name,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "ndim": ndim,
            "edim": edim,
            "ndim_out": ndim_out,
            "num_layers": num_layers,
            "number_neighbors": number_neighbors,
            "activation": activation.__name__,
            "dropout": dropout,
            "residual": residual,
            "multi_class": multi_class,
            "aggregation": aggregation,
            # "details": "updating edge features",
            "early_stopping_patience": early_stopping_patience,
            "use_centralities_nfeats": use_centralities_nfeats,
        }

        if graph_type == "flow":
            graph_model = GraphModel(model, criterion, learning_rate, config, model_name,
                                     labels_mapping, weight_decay=weight_decay, using_wandb=using_wandb, norm=False, multi_class=True)
        elif graph_type == "window":
            graph_model = WindowedGraphModel(model, criterion, learning_rate, config, model_name,
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
        if using_wandb:
            wandb.log(logs)
            wandb.finish()
        else:
            trainer.logger.log_metrics(logs, step=trainer.global_step)
    print(f"==>> elapsed: {elapsed}")


if __name__ == "__main__":
    main()
