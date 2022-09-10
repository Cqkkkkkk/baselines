import torch.nn as nn
from torch_geometric.nn import GINConv
import torch.nn.functional as F

class GINNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

        self.conv1 = GINConv(self.linear1)
        self.conv2 = GINConv(self.linear2)

        self.dropout = dropout
       

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)
