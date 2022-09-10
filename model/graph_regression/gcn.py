import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import pdb


class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, out_dim, dropout=0.5):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.lin = nn.Linear(out_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x

if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from torch_geometric.datasets import TUDataset

    dataset = TUDataset(root='../../dataset/TUDataset', name='MUTAG')
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = GCN(num_node_features=7, hidden_dim=64, out_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.L1Loss()

    def train(model):
        model.train()

        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out.squeeze(dim=-1), data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
    
    
    def test(loader):
        model.eval()

        mae = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            mae += criterion(out.squeeze(dim=-1), data.y)
        return mae  


    for epoch in range(100):
        train(model)
        train_acc = test(train_loader)
        test_acc = test(test_loader)

        print('Epoch {}: TrainMAE {:.3f} TestMAE {:.3f}'.format(epoch, train_acc, test_acc))