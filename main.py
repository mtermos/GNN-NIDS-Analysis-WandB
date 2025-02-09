import torch.nn as nn
import os
import pickle
import torch
import warnings
import wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from src.models import EGAT, EGCN, EGRAPHSAGE
from src.lightning_model import GraphModel
from src.lightning_data import GraphDataModule
from src.dataset.dataset_info import datasets

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def main():
    # Hyperparameters
    dataset_name = "cic_ids_2017_5_percent"
    max_epochs = 10
    learning_rate = 1e-3
    weight_decay = 1e-4
    ndim_out = [128, 128]
    num_layers = 2
    number_neighbors = [25, 10]
    activation = F.relu
    dropout = 0.2
    residual = True
    multi_class = True
    aggregation = "mean"

    dataset = datasets[dataset_name]
    dataset_folder = os.path.join("datasets", dataset.name)
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
        device='cuda' if torch.cuda.is_available() else "cpu"
    )

    data_module = GraphDataModule(
        graphs_folder, batch_size=1, **dataset_kwargs)
    data_module.setup()

    ndim = next(iter(data_module.train_dataloader())).ndata["h"].shape[-1]
    edim = next(iter(data_module.train_dataloader())).edata['h'].shape[-1]

    my_models = {
        "e_gcn": EGCN(ndim, edim, ndim_out, num_layers, activation,
                      dropout, residual, num_classes),
        "e_graphsage": EGRAPHSAGE(ndim, edim, ndim_out, num_layers, activation, dropout,
                                  residual, num_classes, number_neighbors, aggregation),
        "e_gat": EGAT(ndim, edim, ndim_out, num_layers, activation, dropout,
                      residual, num_classes)
    }

    criterion = nn.CrossEntropyLoss(data_module.train_dataset.class_weights)

    config = {
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
    }

    for model_name, model in my_models.items():
        graph_model = GraphModel(model, criterion, learning_rate, config, model_name,
                                 labels_mapping, weight_decay=weight_decay, norm=False, multi_class=True)

        wandb_logger = WandbLogger(
            project=f"GNN-Analysis-{dataset.name}",
            name=model_name,
            config=config,
            save_dir=wandb_runs_path
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename="best-val-acc-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            verbose=False,
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=False,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            log_every_n_steps=0,
            callbacks=[checkpoint_callback, early_stopping_callback],
            default_root_dir=logs_folder,
            logger=wandb_logger
        )

        # Train the model
        trainer.fit(graph_model, datamodule=data_module)

        # Test the model
        trainer.test(graph_model, datamodule=data_module,
                     ckpt_path=checkpoint_callback.best_model_path)
        wandb.finish()
        # print(
        #     f"==>> checkpoint_callback.best_model_path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
