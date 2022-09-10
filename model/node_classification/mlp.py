import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.to_dense_adj import to_dense_adj


class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        # x = to_dense_adj(data.edge_index).squeeze(dim=0)
        h = F.relu(self.linear1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.linear2(h)
        return F.log_softmax(h, dim=1)
     

class FAMLP(nn.Module):
    def __init__(self, in_dim_x, in_dim_adj, hidden_dim, out_dim, dropout=0.2, alpha=0):
        super().__init__()
        self.linear1 = nn.Linear(in_dim_x, hidden_dim)
        self.linear2 = nn.Linear(in_dim_adj, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(adj))
        h = self.alpha * h1 + (1 - self.alpha) * h2
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.linear3(h)
        return F.log_softmax(h, dim=1)
