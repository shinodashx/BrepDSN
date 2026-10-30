import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_head


@register_head('brep_dual_head_enhanced')
class Brep_Dual_Head_Enhanced(nn.Module):

    def __init__(self, dim_in, **kwargs):
        super().__init__()
        dim_out = kwargs['dim_node_out']
        
        self.face_head = nn.Sequential(
            nn.Linear(dim_in, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, dim_out)
        )
        
        self.has_aux = kwargs.get('has_aux', False)
        if self.has_aux:
            aux_dim_out = kwargs['aux_dim_edge_out']
            self.edge_head = nn.Sequential(
                nn.Linear(dim_in, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, aux_dim_out)
            )
            
    def forward(self, batch):
        oriGraph, venoGraph = batch
        
        x_pool_upsample = torch.index_select(venoGraph.x_pool, 0, venoGraph.batch)
        face_features = venoGraph.x + x_pool_upsample
        
        face_pred = self.face_head(face_features)
        face_label = venoGraph.y
        
        predictions = [face_pred]
        labels = [face_label]
        
        if self.has_aux:
            edge_features = venoGraph.edge_attr
            edge_pred = self.edge_head(edge_features)
            
            vo_emap_idx = venoGraph.edge_dual_idx.squeeze()
            edge_label = oriGraph.edge_label[vo_emap_idx]
            
            predictions.append(edge_pred)
            labels.append(edge_label)
            
        if len(predictions) == 1:
            return predictions[0], labels[0]
        else:
            return tuple(predictions), tuple(labels) 
