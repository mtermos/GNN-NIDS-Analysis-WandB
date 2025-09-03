import pytorch_lightning as pl
import torch
import timeit
import os
import json
import wandb
import numpy as np
import pandas as pd
from dgl.nn.pytorch import EdgeWeightNorm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from src.utils import NumpyEncoder, calculate_fpr_fnr_with_global, plot_confusion_matrix


class GraphModel(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, config_dict, model_name, labels_mapping, weight_decay=0, using_wandb=False, norm=False, multi_class=False, label_col="Label", class_num_col="Class", batch_size=1, verbose=True, needs_epoch_hook=False):
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
        self.verbose = verbose
        self.needs_epoch_hook = needs_epoch_hook
        self.save_hyperparameters(config_dict)
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

    def on_train_epoch_start(self):
        if self.needs_epoch_hook:
            self.criterion.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

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

        start_time = timeit.default_timer()
        pred = self.forward(graph, node_features, edge_features)
        elapsed = timeit.default_timer() - start_time
        print(f"==>> elapsed: {elapsed}")

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
        self.log(f"{self.test_prefix}_elapsed", elapsed, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        return {"test_loss": loss, "test_acc": test_acc, "test_f1": weighted_f1, "elapsed": elapsed}
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

        if self.verbose:
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


class WindowedGraphModel(pl.LightningModule):
    def __init__(self, model, criterion, learning_rate, config_dict, model_name, labels_mapping, weight_decay=0, using_wandb=False, norm=False, multi_class=False, label_col="Label", class_num_col="Class", batch_size=1, verbose=True, needs_epoch_hook=False):
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
        self.verbose = verbose
        self.needs_epoch_hook = needs_epoch_hook
        self.save_hyperparameters(config_dict)
        self.train_epoch_metrics = {}
        self.val_epoch_metrics = {}
        self.test_outputs = []
        self.test_prefix = ""
        self.train_outputs = {"preds": [], "targets": []}
        self.val_outputs = {"preds": [], "targets": []}
        self.test_outputs = {"preds": [], "targets": []}

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

    def on_train_epoch_start(self):
        if self.needs_epoch_hook:
            self.criterion.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()
        
    def training_step(self, batch, batch_idx):

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

        self.train_outputs["preds"].extend(preds)
        self.train_outputs["targets"].extend(targets)

        train_acc = (preds == targets).float().mean().item() * 100

        self.log('train_loss', loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', train_acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        return loss

    def on_train_epoch_end(self):

        all_preds = torch.cat(
            [t.unsqueeze(0) for t in self.train_outputs["preds"]]
        ).detach().cpu().numpy()

        all_targets = torch.cat(
            [t.unsqueeze(0) for t in self.train_outputs["targets"]]
        ).detach().cpu().numpy()

        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("train_f1_score", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.train_outputs = {"preds": [], "targets": []}

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
        self.log('val_loss', loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', val_acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.val_outputs["preds"].extend(preds)
        self.val_outputs["targets"].extend(targets)

        # print(
        #     f'==>> self.val_outputs["targets"].shape: {len(self.val_outputs["targets"])}')
        return {"val_loss": loss, "val_acc": val_acc}

    def on_validation_epoch_end(self):
        if getattr(self.trainer, "sanity_checking", False):
            self.val_outputs = {"preds": [], "targets": []}
            return  # skip any summary/logging during the sanity‐check
        all_preds = torch.cat(
            [t.unsqueeze(0) for t in self.val_outputs["preds"]]
        ).detach().cpu().numpy()
        # all_preds = torch.cat(self.val_outputs["preds"]).detach().cpu().numpy()
        all_targets = torch.cat(
            [t.unsqueeze(0) for t in self.val_outputs["targets"]]
        ).detach().cpu().numpy()
        # print(f"==>> all_targets: {all_targets}")
        # all_targets = torch.cat(
        #     self.val_outputs["targets"]).detach().cpu().numpy()
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("val_f1_score", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        report = classification_report(
            all_targets, all_preds, digits=4, output_dict=False, zero_division=0)

        print("Validation Classification Report:\n", report)

        self.val_outputs = {"preds": [], "targets": []}

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

        start_time = timeit.default_timer()
        pred = self.forward(graph, node_features, edge_features)
        elapsed = timeit.default_timer() - start_time
        # print(f"==>> elapsed: {elapsed}")

        loss = self.criterion(pred[test_mask], edge_labels[test_mask])

        preds = pred[test_mask].argmax(dim=1).detach().cpu()
        targets = edge_labels[test_mask].detach().cpu()

        test_acc = (preds == targets).float().mean().item() * 100

        self.log("test_loss", loss, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        self.log("test_acc", test_acc, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        self.test_outputs["preds"].extend(preds)
        self.test_outputs["targets"].extend(targets)
        return {"test_loss": loss, "test_acc": test_acc, "preds": pred, "targets": targets}

    def on_test_epoch_end(self):

        all_preds = torch.cat(
            [t.unsqueeze(0) for t in self.test_outputs["preds"]]
        ).detach().cpu().numpy()

        all_targets = torch.cat(
            [t.unsqueeze(0) for t in self.test_outputs["targets"]]
        ).detach().cpu().numpy()
        # all_targets = torch.cat(
        #     self.test_outputs["targets"]).detach().cpu().numpy()
        self.test_outputs = {"preds": [], "targets": []}
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100.0
        self.log("test_f1", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)

        all_targets = np.vectorize(self.labels_mapping.get)(all_targets)
        all_preds = np.vectorize(self.labels_mapping.get)(all_preds)

        cm = confusion_matrix(all_targets, all_preds, labels=self.labels)

        cr = classification_report(
            all_targets, all_preds, digits=4, output_dict=True, zero_division=0)
        report = classification_report(
            all_targets, all_preds, digits=4, output_dict=False, zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds,
                               average="weighted") * 100

        self.log(f"{self.test_prefix}_test_f1", weighted_f1, on_epoch=True,
                 prog_bar=True, batch_size=self.batch_size)
        results_fpr_fnr = calculate_fpr_fnr_with_global(cm)
        fpr = results_fpr_fnr["global"]["FPR"]
        fnr = results_fpr_fnr["global"]["FNR"]

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

        # cm_normalized = confusion_matrix(
        #     all_targets, all_preds, labels=self.labels, normalize="true")
        # fig = plot_confusion_matrix(cm=cm,
        #                             normalized=False,
        #                             target_names=self.labels,
        #                             title=f"Confusion Matrix of {self.model_name}",
        #                             file_path=None,
        #                             show_figure=False)

        # if self.using_wandb:
        #     wandb.log({f"confusion_matrix_{self.model_name}": wandb.Image(
        #         fig), "epoch": self.current_epoch})
        # fig = plot_confusion_matrix(cm=cm_normalized,
        #                             normalized=True,
        #                             target_names=self.labels,
        #                             title=f"Confusion Matrix of {self.model_name}",
        #                             file_path=None,
        #                             show_figure=False)

        # if self.using_wandb:
        #     wandb.log({f"confusion_matrix_{self.model_name}_normalized": wandb.Image(
        #         fig), "epoch": self.current_epoch})

    def configure_optimizers(self):
        """
        Configure the optimizer. In this example, we use Adam.
        """
        optimizer = torch.optim.Adam(self.model.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        return optimizer
