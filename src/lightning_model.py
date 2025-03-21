import pytorch_lightning as pl
import torch as th
import timeit
import os
import json
import wandb
import numpy as np
from dgl.nn.pytorch import EdgeWeightNorm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
# from src.utils import NumpyEncoder, calculate_fpr_fnr_with_global, plot_confusion_matrix

import itertools

import matplotlib.pyplot as plt


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
    else:
        plt.close(fig)

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
    def __init__(self, model, criterion, learning_rate, config, model_name, labels_mapping, weight_decay=0, using_wandb=False, norm=False, multi_class=False, label_col="Label", class_num_col="Class", batch_size=1, verbose=True):
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
            edge_weight = th.ones(
                graph.num_edges(), dtype=th.float32, device=graph.device)
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
        optimizer = th.optim.Adam(self.model.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        return optimizer
