import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.graphgym.register import register_stage, register_network



@register_stage('DUAL_stack')
class GNNStackStage(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_layers, **kwargs):
        super().__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = self.GNNLayer(d_in, dim_out, **kwargs)
            self.add_module(f'layer{i}', layer)

    def GNNLayer(self, dim_in: int, dim_out: int, has_act: bool = False, **kwargs):
        r"""Creates a GNN layer, given the specified input and output dimensions
        and the underlying configuration in :obj:`cfg`.

        Args:
            dim_in (int): The input dimension
            dim_out (int): The output dimension.
            has_act (bool, optional): Whether to apply an activation function
                after the layer. (default: :obj:`True`)
        """
        return register.layer_dict[cfg.gnn.layer_type](
            layer_config=new_layer_config(
                dim_in,
                dim_out,
                1,
                has_act=has_act,
                has_final_act=False,
                has_bias=not cfg.gnn.batchnorm,
                cfg=cfg,
            ),
            **kwargs
        )

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return tuple(batch)



@register_network('DUAL_GNN_BASE')
class Dual_GNN_base(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        GNNHead = register.head_dict[cfg.gnn.head]

        encoder_kwargs = dict(cfg.share)
        NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
        self.encoder = NodeEncoder(cfg.gnn.dim_inner, **encoder_kwargs)
        dim_in = cfg.gnn.dim_inner

        GNN_kwargs = dict(cfg.gnn.layer_cfg)
        self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp, **GNN_kwargs)

        head_kwargs = dict(cfg.gnn.head_cfg)
        head_kwargs.update(encoder_kwargs)
        head_kwargs['has_aux'] = cfg.model.has_aux
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, **head_kwargs)

        self.apply(init_weights)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch