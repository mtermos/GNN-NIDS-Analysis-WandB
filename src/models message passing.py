import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl


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
            score = self.W(th.cat([h_u, h_v, h_uv], 1))
        else:
            score = self.W(th.cat([h_u, h_v], 1))

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
                th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
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
                th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
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
        return {"e": self.activation(self.attn_fc(th.cat([edges.src["h"], edges.dst["h"]], dim=2)))}

    def message_func(self, edges):
        return {"m": edges.data['h'], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        z = th.sum(alpha * nodes.mailbox['m'], dim=1)
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
                th.cat([g.ndata['h'], g.ndata['z']], 2)))
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
