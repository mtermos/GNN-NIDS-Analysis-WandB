import os
import pickle

import dgl
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from dgl import from_networkx
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Dataset

# Customize this dataset class to encapsulate your graph-loading and processing.


class WindowedGraphDataset(Dataset):
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

        self.labels = []

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

        self.graphs = self._load_and_process_graph()

        self.class_weights = self._compute_class_weights()

    def _load_and_process_graph(self):
        # Determine the filename based on the split
        graphs_dir = os.path.join(self.graphs_folder, self.split, "graphs")
        graph_files = sorted([
            f for f in os.listdir(graphs_dir) if f.startswith("graph_") and f.endswith(".pkl")
        ], key=lambda x: int(x.split('_')[1].split('.')[0]))

        edge_attributes = ['h', self.label_col, self.class_num_col]
        graphs = []
        for fname in graph_files:
            with open(os.path.join(graphs_dir, fname), 'rb') as f:
                G_nx = pickle.load(f)
                if self.use_node_features:
                    G = from_networkx(G_nx, edge_attrs=edge_attributes,
                                      node_attrs=["n_feats"]).to(self.device)
                else:
                    G = from_networkx(
                        G_nx, edge_attrs=edge_attributes).to(self.device)

                # Get the number of features from the edge feature "h"
                num_features = G.edata['h'].shape[1]

                # If masking is used in training, filter out the masked edges.
                if self.using_masking and self.split == 'training':
                    # Assumes you want to mask based on the "class_num" attribute.
                    training_mask = G.edata[self.class_num_col] != self.masked_class
                    G = dgl.edge_subgraph(G, training_mask)

                # Process node features:
                if self.use_node_features:
                    # For example, you might want to use the “n_feats” field.
                    # G.ndata["h"] = th.cat([G.ndata["n_feats"].to(self.device), th.ones(
                    #     G.num_nodes(), num_features, device=self.device)], dim=1)
                    G.ndata["h"] = G.ndata["n_feats"].to(self.device)
                else:
                    # Otherwise, initialize node features as ones.
                    G.ndata['h'] = th.ones(
                        G.num_nodes(), num_features, device=self.device)

                # Reshape node and edge features if required by your model.
                G.ndata['h'] = G.ndata['h'].reshape(
                    G.ndata['h'].shape[0], 1, -1)
                G.edata['h'] = G.edata['h'].reshape(
                    G.edata['h'].shape[0], 1, -1)

                # Create masks in the edge data for later usage in training/validation/testing.
                if self.split == 'training':
                    G.edata['train_mask'] = th.ones(
                        G.edata['h'].shape[0], dtype=th.bool, device=self.device)
                elif self.split == 'validation':
                    G.edata['val_mask'] = th.ones(
                        G.edata['h'].shape[0], dtype=th.bool, device=self.device)
                elif self.split == 'testing':
                    G.edata['test_mask'] = th.ones(
                        G.edata['h'].shape[0], dtype=th.bool, device=self.device)

                graphs.append(G)

        return graphs
        # return G

    def _compute_class_weights(self):
        all_targets = []

        for g in self.graphs:
            if self.multi_class:
                targets = g.edata[self.class_num_col].cpu().numpy()
            else:
                targets = g.edata[self.label_col].cpu().numpy()

            if self.using_masking and self.split == 'training':
                mask = (targets != self.masked_class)
                targets = targets[mask]

            all_targets.extend(targets)

        # Convert to numpy array
        all_targets = np.array(all_targets)

        # Get unique classes and compute weights
        classes = np.unique(all_targets)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=all_targets
        )

        return th.FloatTensor(weights)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return g


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
        # using the edge attribute names “label” or “class_num”. Adjust as necessary.
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
            # For example, you might want to use the “n_feats” field.
            # G.ndata["h"] = th.cat([G.ndata["n_feats"].to(self.device), th.ones(
            #     G.num_nodes(), num_features, device=self.device)], dim=1)
            G.ndata["h"] = G.ndata["n_feats"].to(self.device)
        else:
            # Otherwise, initialize node features as ones.
            G.ndata['h'] = th.ones(
                G.num_nodes(), num_features, device=self.device)

        # Reshape node and edge features if required by your model.
        G.ndata['h'] = G.ndata['h'].reshape(G.ndata['h'].shape[0], 1, -1)
        G.edata['h'] = G.edata['h'].reshape(G.edata['h'].shape[0], 1, -1)

        # Create masks in the edge data for later usage in training/validation/testing.
        if self.split == 'training':
            G.edata['train_mask'] = th.ones(
                G.edata['h'].shape[0], dtype=th.bool, device=self.device)
        elif self.split == 'validation':
            G.edata['val_mask'] = th.ones(
                G.edata['h'].shape[0], dtype=th.bool, device=self.device)
        elif self.split == 'testing':
            G.edata['test_mask'] = th.ones(
                G.edata['h'].shape[0], dtype=th.bool, device=self.device)

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
        return th.FloatTensor(weights)

    def __len__(self):
        # If you are loading one complete graph per split, the dataset length is 1.
        # If you have many graphs stored in a single file, you would change this.
        return 1

    def __getitem__(self, idx):
        # Since there is only one graph, simply return it.
        # (For a dataset with multiple graphs, return the idx-th graph.)
        return self.graph

# Now create a LightningDataModule that wraps your dataset for train, val, and test.


# train_dataset = WindowedGraphDataset(train_graphs, train_labels)
# train_loader = GraphDataLoader(train_dataset, batch_size=1, shuffle=True)

# for batched_graph, labels in train_loader:
#     batched_graph = batched_graph.to(device)
#     labels = labels.to(device)

#     pred = model(batched_graph)
#     loss = loss_fn(pred, labels)
#     # Backprop, etc.

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, graphs_folder, graph_type, batch_size=1, **dataset_kwargs):
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
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.dataset_kwargs = dataset_kwargs

        if self.graph_type == "flow":
            self.dataset_class = CustomGraphDataset
            self.collate_fn = lambda x: x[0]
        if self.graph_type == "window":
            self.dataset_class = WindowedGraphDataset
            self.collate_fn = lambda x: x[0]

    def setup(self, stage=None):

        # For the 'fit' stage, load train and validation splits.
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                self.graphs_folder, split='training', **self.dataset_kwargs)
            self.val_dataset = self.dataset_class(
                self.graphs_folder, split='validation', **self.dataset_kwargs)
        # For the 'test' stage, load the test split.
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_class(
                self.graphs_folder, split='testing', **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.dataset_kwargs["num_workers"],
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.dataset_kwargs["num_workers"],
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.dataset_kwargs["num_workers"],
                          collate_fn=self.collate_fn)
