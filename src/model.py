"""Graph neural network models for QM9 property prediction."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import NNConv, global_mean_pool


class MPNN(nn.Module):
    """Simple message-passing neural network with edge-conditioned convolutions."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.node_emb = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        edge_net_1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        edge_net_2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        edge_net_3 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

        self.conv1 = NNConv(hidden_dim, hidden_dim, edge_net_1, aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim, edge_net_2, aggr="mean")
        self.conv3 = NNConv(hidden_dim, hidden_dim, edge_net_3, aggr="mean")

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        batch = data.batch

        x = self.node_emb(x)

        h = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = x + h

        h = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = x + h

        h = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = x + h

        graph_emb = global_mean_pool(x, batch)
        return self.readout(graph_emb)
