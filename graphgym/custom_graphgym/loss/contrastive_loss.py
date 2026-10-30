import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import register_loss
from torch_geometric.graphgym.config import cfg
import torch_scatter
from torch_geometric.utils import degree


@register_loss('supervised_face_vertices_congruence')
class SupervisedFaceVerticesCongruenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = cfg.model.contrastive_loss_cfg.get('temperature', 0.1)

        gnn_dim = cfg.gnn.dim_inner
        projection_dim = gnn_dim // 2

        self.face_proj_head = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim), nn.ReLU(), nn.Linear(gnn_dim, projection_dim)
        )
        self.vertex_proj_head = nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim), nn.ReLU(), nn.Linear(gnn_dim, projection_dim)
        )

    def _aggregate_boundary_vertex_features(self, oriGraph, venoGraph, vertex_features):
        face_indices_long = venoGraph.edge_index[0]
        face_indices_short = venoGraph.edge_index[1]
        face_indices = torch.cat([face_indices_long, face_indices_short], dim=0)

        num_veno_edges = venoGraph.edge_index.size(1)
        veno_edge_ids = torch.arange(num_veno_edges, device=vertex_features.device).repeat(2)

        valid_mask = venoGraph.edge_dual_idx.squeeze() >= 0
        face_indices = face_indices[valid_mask.repeat(2)]
        veno_edge_ids = veno_edge_ids[valid_mask.repeat(2)]

        if face_indices.numel() == 0:
            return None, None

        ori_edge_ids = venoGraph.edge_dual_idx.squeeze()[veno_edge_ids]

        boundary_vertex_indices = oriGraph.edge_index[:, ori_edge_ids].t().reshape(-1)
        face_indices_for_vertices = face_indices.repeat_interleave(2)

        face_degrees = degree(face_indices_for_vertices, num_nodes=venoGraph.num_nodes)

        aggregated_boundary_features = torch_scatter.scatter_sum(
            vertex_features[boundary_vertex_indices], face_indices_for_vertices,
            dim=0, dim_size=venoGraph.num_nodes
        )

        aggregated_boundary_features = aggregated_boundary_features / (face_degrees.unsqueeze(-1) + 1e-8)

        valid_faces_mask = face_degrees > 0
        return aggregated_boundary_features, valid_faces_mask

    def _compute_supcon_loss(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        labels_matrix = labels.contiguous().view(-1, 1)

        positives_mask = torch.eq(labels_matrix, labels_matrix.T).float().to(features.device)
        positives_mask.fill_diagonal_(0)

        sim_matrix = torch.matmul(features, features.t()) / self.temperature
        logits_mask = torch.ones_like(positives_mask).fill_diagonal_(0)

        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        num_positives = positives_mask.sum(1)
        has_positives = num_positives > 0

        mean_log_prob_pos = (positives_mask * log_prob).sum(1)
        mean_log_prob_pos[has_positives] /= num_positives[has_positives]

        loss = -mean_log_prob_pos[has_positives]

        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=features.device)

    def forward(self, batch_data):
        oriGraph, venoGraph = batch_data

        face_features = self.face_proj_head(venoGraph.x)
        vertex_features = self.vertex_proj_head(oriGraph.x)

        aggregated_boundary_features, valid_faces_mask = self._aggregate_boundary_vertex_features(
            oriGraph, venoGraph, vertex_features
        )

        face_view_features = face_features[valid_faces_mask]
        boundary_view_features = aggregated_boundary_features[valid_faces_mask]
        labels = venoGraph.y[valid_faces_mask]

        features = torch.cat([face_view_features, boundary_view_features], dim=0)
        labels = labels.repeat(2)

        sup_con_loss = self._compute_supcon_loss(features, labels)

        return sup_con_loss
