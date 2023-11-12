import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)

