import torch
from torch_geometric.nn import NNConv


class my_NNCONV(NNConv):
    def message(self, x_j, edge_attr):
        b, c = x_j.shape
        x_j = x_j.view(b, 4, c // 4).unsqueeze(2)
        weight = self.nn(edge_attr)
        weight = weight.view(b, 4, self.in_channels_l // 4, self.out_channels // 4)
        x_j = torch.einsum('bnxy,bnyz->bnxz', x_j, weight).squeeze(2)
        x_j = x_j.view(b, c)
        return x_j