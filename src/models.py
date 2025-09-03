import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl

# files 1 to 26


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
            h_uv = h_uv.view(h_uv.shape[0], h_uv.shape[-1])
            score = self.W(torch.cat([h_u, h_v, h_uv], 1))
        else:
            score = self.W(torch.cat([h_u, h_v], 1))

        return {'score': score}

    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
            graph.ndata['h'] = nfeats
            graph.edata['h'] = efeats
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


#############################
#############################
#############################
# E_GCN
class GCNLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, norm=True, use_node_h=True):
        super(GCNLayer, self).__init__()

        if use_node_h:
            self.W_apply = nn.Linear(ndim_in + edim, ndim_out)
        else:
            self.W_apply = nn.Linear(edim, ndim_out)

        self.activation = activation
        self.norm = norm
        self.use_node_h = use_node_h

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

            if self.use_node_h:
                g.ndata['h'] = self.activation(self.W_apply(
                    torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            else:
                g.ndata['h'] = self.activation(self.W_apply(g.ndata['h_neigh']))
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
        return self.pred(g, h, efeats)


#############################
#############################
#############################
# E_GraphSAGE Model


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, aggregation, num_neighbors=None, edge_update=False):
        super(SAGELayer, self).__init__()
        self.W_apply = nn.Linear(ndim_in + edim, ndim_out)
        self.activation = activation
        self.aggregation = aggregation
        self.num_neighbors = num_neighbors
        self.edge_update = edge_update
        
        if edge_update:
            self.edge_update_layer = nn.Linear(2 * ndim_out + edim, edim)

        if aggregation == "pool":
            self.pool_fc = nn.Linear(ndim_out, ndim_out)
        elif aggregation == "lstm":
            self.lstm = nn.LSTM(ndim_out, ndim_out, batch_first=True)

    def message_func(self, edges):
        # if multi_graph then the node features of the source node are repeated
        # after concatenation, for each edge, we have [src_nfeats_1 , ... , src_nfeats_n, efeats_1, ... efeats_m]
        # after that we apply linear layer to create new featurescset called m.
        return {'m': edges.data['h']}

    def update_edge_features(self, edges):
        cat_input = torch.cat(
            [edges.src["h"], edges.dst["h"], edges.data["h"]], dim=-1)
        updated_edge = self.activation(self.edge_update_layer(cat_input))
        return {"h": updated_edge}

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
            g.ndata['h'] = self.activation(self.W_apply(
                torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            
            if self.edge_update:
                g.apply_edges(self.update_edge_features)
                return g.ndata['h'], g.edata['h']
            else:
                return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, aggregation, dropout, num_neighbors, edge_update=False):
        super(SAGE, self).__init__()
        self.edge_update = edge_update
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    SAGELayer(ndim_in, edim, ndim_out[layer], activation, aggregation, num_neighbors[layer] if num_neighbors else None, edge_update))
            else:
                self.layers.append(SAGELayer(
                    ndim_out[layer-1], edim, ndim_out[layer], activation, aggregation, num_neighbors[layer] if num_neighbors else None, edge_update))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):

        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
                if self.edge_update:
                    efeats = self.dropout(efeats)
            
            if self.edge_update:
                nfeats, efeats = layer(g, nfeats, efeats)
            else:
                nfeats = layer(g, nfeats, efeats)
        
        if self.edge_update:
            return nfeats.sum(1), efeats.sum(1)
        else:
            return nfeats.sum(1)


class EGRAPHSAGE(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=True, num_class=2, num_neighbors=None, aggregation="mean", edge_update=False):
        super().__init__()
        self.edge_update = edge_update
        self.gnn = SAGE(ndim_in, edim, ndim_out, num_layers,
                        activation, aggregation, dropout, num_neighbors, edge_update)
        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        if self.edge_update:
            h, e = self.gnn(g, nfeats, efeats)
            return self.pred(g, h, e)
        else:
            h = self.gnn(g, nfeats, efeats)
            return self.pred(g, h, efeats)

#############################
#############################
#############################
# E_GAT Model


class GATLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation, num_neighbors=None, edge_update=False):
        super(GATLayer, self).__init__()
        self.W_apply = nn.Linear(ndim_in + edim, ndim_out)
        self.attn_fc = nn.Linear(2*ndim_in, 1)
        self.activation = activation
        self.num_neighbors = num_neighbors
        self.edge_update = edge_update
        
        if edge_update:
            self.edge_update_layer = nn.Linear(2 * ndim_out + edim, edim)

    def edge_attention(self, edges):
        return {"e": self.activation(self.attn_fc(torch.cat([edges.src["h"], edges.dst["h"]], dim=2)))}

    def update_edge_features(self, edges):
        cat_input = torch.cat(
            [edges.src["h"], edges.dst["h"], edges.data["h"]], dim=-1)
        updated_edge = self.activation(self.edge_update_layer(cat_input))
        return {"h": updated_edge}

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
            
            if self.edge_update:
                g.apply_edges(self.update_edge_features)
                return g.ndata['h'], g.edata['h']
            else:
                return g.ndata['h']


class GAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout, num_neighbors, edge_update=False):
        super().__init__()
        self.edge_update = edge_update
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.layers.append(
                    GATLayer(ndim_in, edim, ndim_out[layer], activation, num_neighbors[layer] if num_neighbors else None, edge_update))
            else:
                self.layers.append(
                    GATLayer(ndim_out[layer-1], edim, ndim_out[layer], activation, num_neighbors[layer] if num_neighbors else None, edge_update))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
                if self.edge_update:
                    efeats = self.dropout(efeats)
            
            if self.edge_update:
                nfeats, efeats = layer(g, nfeats, efeats)
            else:
                nfeats = layer(g, nfeats, efeats)
        
        if self.edge_update:
            return nfeats.sum(1), efeats.sum(1)
        else:
            return nfeats.sum(1)


class EGAT(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=False, num_class=2, num_neighbors=None, edge_update=False):
        super().__init__()
        self.edge_update = edge_update
        self.gnn = GAT(ndim_in, edim, ndim_out, num_layers,
                       activation, dropout, num_neighbors, edge_update)

        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        if self.edge_update:
            h, e = self.gnn(g, nfeats, efeats)
            return self.pred(g, h, e)
        else:
            h = self.gnn(g, nfeats, efeats)
            return self.pred(g, h, efeats)


#############################
#############################
#############################
# E_GIN Model

class GINLayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ndim_in + edim, ndim_out),
            nn.ReLU(),
            nn.Linear(ndim_out, ndim_out),
            nn.ReLU(),
        )

    def forward(self, g, nfeats, efeats):
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(fn.copy_e('h', 'm'), fn.sum('m', 'h_neigh'))
            h_input = torch.cat([g.ndata['h'], g.ndata['h_neigh']], dim=-1)
            g.ndata['h'] = self.mlp(h_input)
            return g.ndata['h']


class GIN(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = ndim_in if i == 0 else ndim_out[i - 1]
            self.layers.append(GINLayer(in_dim, edim, ndim_out[i], activation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i > 0:
                nfeats = self.dropout(nfeats)
                if self.edge_update:
                    efeats = self.dropout(efeats)
            
            if self.edge_update:
                nfeats, efeats = layer(g, nfeats, efeats)
            else:
                nfeats = layer(g, nfeats, efeats)
        if self.edge_update:
            return nfeats.sum(1), efeats.sum(1)
        else:
            return nfeats.sum(1)


class EGIN(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, num_layers=2, activation=F.relu, dropout=0.2, residual=True, num_class=2, edge_update=False):
        super().__init__()
        self.edge_update = edge_update
        self.gnn = GIN(ndim_in, edim, ndim_out,
                       num_layers, activation, dropout)
        self.pred = MLPPredictor(
            ndim_out[-1], edim, num_class, activation, residual)

    def forward(self, g, nfeats, efeats):
        if self.edge_update:
            h, e = self.gnn(g, nfeats, efeats)
            return self.pred(g, h, e)
        else:
            h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h, efeats)


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
