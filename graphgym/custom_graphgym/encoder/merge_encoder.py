import torch
import torch.nn as nn

from torch_geometric.graphgym.register import register_node_encoder

class Brep_Conv2D_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Brep_Conv2D_Block, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x



@register_node_encoder('dual_node_brep_f')
class DualNodeEncoder_Brep_F(torch.nn.Module):
    """
    An elegant and effective B-Rep encoder designed for academic publication (e.g., NIPS).
    It captures rich edge information using a lightweight PointNet-inspired architecture,
    replacing complex statistical pooling with a more principled and learnable approach.
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__()

        # 1. Face Node Encoder (remains unchanged)
        # Uses a 2D CNN to process face image-like features.
        self.venoNodeEmbed = Brep_Conv2D_Block(in_channels=kwargs['dim_node_in'], out_channels=embed_dim)

        # 2. Edge Encoder (PointNet-inspired design)
        edge_dim = kwargs['dim_edge_in']

        # Point-wise MLP (implemented with 1x1 Conv) to lift point features on the edge curve
        self.point_mlp = nn.Sequential(
            nn.Conv1d(edge_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
        )

        # MLP to process aggregated spatial features (from mean & max pooling)
        self.edge_spatial_embed = nn.Sequential(
            nn.Linear(128 * 2, embed_dim, bias=False), # 128 for mean, 128 for max
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(inplace=True)
        )

        # Edge type encoder (remains unchanged)
        self.venoEdgeTypeEmbed = nn.Linear(edge_dim, embed_dim)

        # Final fusion layer for spatial and type features
        self.edge_fusion = nn.Linear(embed_dim * 2, embed_dim)

        # 3. Original Graph Vertex Encoder (remains unchanged)
        # Uses sinusoidal positional encoding for vertex coordinates.
        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * torch.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)
        self.origNodeEmbed = nn.Linear(self.embedding_dim + 3, embed_dim)

    def forward(self, batch):
        oriGraph, venoGraph = batch

        # Encode face nodes
        venoGraph.x = self.venoNodeEmbed(venoGraph.x)

        # --- Encode Edges ---
        edge_attr_tensor = venoGraph.edge_attr
        edge_spatial_feat = edge_attr_tensor[:, :, :-1]  # Shape: (E, C, L_spatial)
        edge_type_feat = edge_attr_tensor[:, :, -1]      # Shape: (E, C)

        # 1. Extract spatial features from edge curves
        point_features = self.point_mlp(edge_spatial_feat)  # (E, 128, L_spatial)

        # 2. Aggregate point features using mean and max pooling
        mean_feat = torch.mean(point_features, dim=2)  # (E, 128)
        max_feat = torch.max(point_features, dim=2)[0] # (E, 128)
        aggregated_feat = torch.cat([mean_feat, max_feat], dim=1)  # (E, 256)
        spatial_embedding = self.edge_spatial_embed(aggregated_feat) # (E, embed_dim)

        # 3. Encode edge type
        type_embedding = self.venoEdgeTypeEmbed(edge_type_feat) # (E, embed_dim)

        # 4. Fuse spatial and type embeddings
        combined_embedding = torch.cat([spatial_embedding, type_embedding], dim=1)
        venoGraph.edge_attr = self.edge_fusion(combined_embedding)
        venoGraph.x_pool = 0

        # Encode original graph vertices
        projections = torch.einsum('bd,de->be', oriGraph.x, self.basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=1)
        oriGraph.x = self.origNodeEmbed(torch.cat([embeddings, oriGraph.x], dim=1))

        return oriGraph, venoGraph
