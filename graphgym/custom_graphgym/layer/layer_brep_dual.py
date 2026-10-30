import torch.nn as nn
import torch
import torch.nn.functional as F

from torch_geometric.graphgym.register import register_layer, layer_dict
from torch_geometric.nn import MLP, max_pool_x
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.resolver import normalization_resolver
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch
from .layer_uvnet import my_NNCONV


@register_layer('Brep_Veno')
class Brep_Veno(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        dim_in, dim_out = layer_config.dim_in, layer_config.dim_out
        edge_func = nn.Linear(layer_config.dim_in, layer_config.dim_in * layer_config.dim_out // 4)
        self.conv = my_NNCONV(dim_in, dim_out, edge_func, bias=False)

        self.mlp = MLP(
            in_channels=dim_in, hidden_channels=dim_in * 2, out_channels=dim_out,
            num_layers=2, plain_last=True
        )
        self.edge_proj = nn.Linear(dim_in, dim_in)
        self.edge_conv = MLP(
            in_channels=dim_in, hidden_channels=dim_in * 2, out_channels=dim_out,
            num_layers=2, plain_last=True
        )
        self.n_norm = nn.BatchNorm1d(dim_out)
        self.n_act = nn.LeakyReLU()
        self.n_param = nn.Parameter(torch.FloatTensor([0.0]))
        self.e_norm = nn.BatchNorm1d(dim_out)
        self.e_act = nn.LeakyReLU()
        self.e_param = nn.Parameter(torch.FloatTensor([0.0]))

        self.global_linear = nn.Linear(dim_in, dim_out)

    def forward(self, graph):
        h = (1 + self.n_param) * graph.x
        h = self.conv(h, graph.edge_index, graph.edge_attr)
        h = self.mlp(h)
        h = self.n_act(self.n_norm(h))

        e_src, e_dst = graph.edge_index[0], graph.edge_index[1]
        proj1, proj2 = self.edge_proj(h[e_src]), self.edge_proj(h[e_dst])
        agg = proj1 + proj2 + graph.dual_attr
        eh = self.edge_conv((1 + self.e_param) * graph.edge_attr + agg)
        eh = self.e_act(self.e_norm(eh))

        graph.x = h
        graph.edge_attr = eh

        pool_h, pool_batch = max_pool_x(graph.batch, graph.x, graph.batch)
        graph.x_pool = graph.x_pool + F.dropout(self.global_linear(pool_h))

        return graph


@register_layer('Brep_Orig')
class Brep_Orig(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        dim_in = layer_config.dim_in
        self.heads = kwargs['o_node_heads']
        self.dropout = kwargs['o_node_dropout']

        self.attn1 = PerformerAttention(channels=dim_in, heads=self.heads)
        self.attn2 = PerformerAttention(channels=dim_in, heads=self.heads)

        norm = kwargs['o_node_norm']
        self.norm1 = normalization_resolver(norm, dim_in)
        self.norm2 = normalization_resolver(norm, dim_in)
        self.norm3 = normalization_resolver(norm, dim_in)

    def forward(self, graph):
        x, batch = graph.x, graph.batch

        h1 = self.norm1(x)
        h1, mask = to_dense_batch(h1, batch)
        h1 = self.attn1(h1, mask=mask)
        h1 = h1[mask]
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        x = h1 + x

        h1 = self.norm2(x)
        h1, mask = to_dense_batch(h1, batch)
        h1 = self.attn2(h1, mask=mask)
        h1 = h1[mask]
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        x = h1 + x

        x = self.norm3(x)
        graph.x = x
        return graph


@register_layer('Brep_Veno_to_Orig')
class Brep_Veno_To_Orig(MessagePassing):
    def __init__(self, layer_config, **kwargs):
        super(Brep_Veno_To_Orig, self).__init__()

    def forward(self, orig_graph, veno_graph):
        edge_attr = veno_graph.edge_attr
        ones_padding = torch.ones((1, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, ones_padding], dim=0)
        ov_edge_map_idx = orig_graph.edge_dual_idx.squeeze()
        orig_graph.dual_attr = edge_attr[ov_edge_map_idx]

        x_structural = self.aggregate(orig_graph.dual_attr, orig_graph.edge_index[1])
        orig_graph.x = orig_graph.x + x_structural
        return orig_graph


@register_layer('Brep_Orig_to_Veno')
class Brep_Orig_To_Veno(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super(Brep_Orig_To_Veno, self).__init__()
        dim_in = layer_config.dim_in
        self.edge_proj = nn.Linear(dim_in, dim_in)

    def forward(self, orig_graph, veno_graph):
        h, edge_index = orig_graph.x, orig_graph.edge_index
        e_src, e_dst = self.edge_proj(h[edge_index[0]]), self.edge_proj(h[edge_index[1]])
        orig_e_attr = e_src + e_dst
        vo_edge_map_idx = veno_graph.edge_dual_idx.squeeze()
        veno_graph.dual_attr = orig_e_attr[vo_edge_map_idx]
        return veno_graph


@register_layer("Brep_Veno_Orig_Layer")
class Brep_Veno_Orig_Layer(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        self.veno_stage = layer_dict[kwargs['veno_stage']](layer_config, **kwargs)
        self.orig_stage = layer_dict[kwargs['orig_stage']](layer_config, **kwargs)
        self.veno_2_orig = layer_dict[kwargs['veno_2_orig']](layer_config, **kwargs)
        self.orig_2_veno = layer_dict[kwargs['orig_2_veno']](layer_config, **kwargs)

    def forward(self, batch):
        origGraph, venoGraph = batch
        origGraph = self.veno_2_orig(origGraph, venoGraph)
        origGraph = self.orig_stage(origGraph)
        venoGraph = self.orig_2_veno(origGraph, venoGraph)
        venoGraph = self.veno_stage(venoGraph)
        return origGraph, venoGraph
