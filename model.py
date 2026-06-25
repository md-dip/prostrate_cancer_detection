"""
HybridProstateCancerNet — exact architecture from the trained notebook.
Triple-branch: EfficientNet-B4 (512) + GAT (256) + GCN (64) → SE-gated fusion (832) → classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


class HybridProstateCancerNet(nn.Module):
    def __init__(self, num_classes=2, hidden=256, gat_heads=4, dropout=0.4):
        super().__init__()
        H = hidden

        # Branch 1: EfficientNet-B4
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False,
            num_classes=0, global_pool='avg'
        )
        cnn_dim = self.backbone.num_features  # 1792

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Branch 2: GAT
        self.gat_node_proj = nn.Sequential(nn.Linear(cnn_dim, H), nn.ReLU())
        self.gat1 = GATConv(H, H // gat_heads, heads=gat_heads, dropout=0.2)
        self.gat2 = GATConv(H, H // gat_heads, heads=gat_heads, dropout=0.2)
        self.gat_norm1 = nn.LayerNorm(H)
        self.gat_norm2 = nn.LayerNorm(H)

        # Branch 3: 3-layer GCN on SLIC graph
        self.gcn_input_proj = nn.Sequential(nn.Linear(8, H), nn.ReLU())
        self.gcn1 = GCNConv(H, H)
        self.gcn2 = GCNConv(H, H // 2)
        self.gcn3 = GCNConv(H // 2, H // 4)
        self.gcn_norm1 = nn.LayerNorm(H)
        self.gcn_norm2 = nn.LayerNorm(H // 2)
        self.gcn_norm3 = nn.LayerNorm(H // 4)

        # SE-gated fusion: 512 + 256 + 64 = 832
        fusion_dim = 512 + H + H // 4  # 832
        self.se_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 8),
            nn.ReLU(),
            nn.Linear(fusion_dim // 8, fusion_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, imgs, graph_batch):
        B = imgs.size(0)

        cnn_feats = self.backbone(imgs)
        cnn_out = self.cnn_proj(cnn_feats)

        gat_nodes = self.gat_node_proj(cnn_feats)
        src = torch.arange(B, device=imgs.device).repeat_interleave(B)
        dst = torch.arange(B, device=imgs.device).repeat(B)
        mask = src != dst
        ei_gat = torch.stack([src[mask], dst[mask]])
        bv_gat = torch.arange(B, device=imgs.device)

        x = self.gat_norm1(F.elu(self.gat1(gat_nodes, ei_gat)))
        x = self.gat_norm2(F.elu(self.gat2(x, ei_gat)))
        gat_out = global_mean_pool(x, bv_gat)

        if graph_batch is not None:
            gx = self.gcn_input_proj(graph_batch.x)
            ei = graph_batch.edge_index
            gb = graph_batch.batch
            gx = self.gcn_norm1(F.relu(self.gcn1(gx, ei)))
            gx = self.gcn_norm2(F.relu(self.gcn2(gx, ei)))
            gx = self.gcn_norm3(F.relu(self.gcn3(gx, ei)))
            gcn_out = global_mean_pool(gx, gb)
        else:
            gcn_out = torch.zeros(B, self.gcn_norm3.normalized_shape[0],
                                  device=imgs.device)

        fused = torch.cat([cnn_out, gat_out, gcn_out], dim=1)
        gate = self.se_gate(fused)
        fused = fused * gate
        return self.classifier(fused)
