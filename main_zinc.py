import torch
import pdb
import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
from model.graph_regression.gcn import GCN


from utils import Int2Float


train_dataset = ZINC(root='./dataset/zinc', subset=True, split='train', pre_transform=Int2Float())
val_dataset = ZINC(root='./dataset/zinc', subset=True, split='val', pre_transform=Int2Float())
test_dataset = ZINC(root='./dataset/zinc', subset=True, split='test', pre_transform=Int2Float())


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = GCN(num_node_features=train_dataset.num_features, hidden_dim=64, out_dim=16)
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