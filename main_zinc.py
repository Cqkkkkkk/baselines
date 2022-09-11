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


def train(model, loader):
    model.train()
    
    for data in loader:   # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out.squeeze(dim=-1), data.y)  
        loss.backward()                   
        optimizer.step() 
        optimizer.zero_grad()  


def test(model, loader):
    model.eval()

    mae = 0
    for data in loader:  # Iterate in batches over the test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        mae += criterion(out.squeeze(dim=-1), data.y)
    return mae  


for epoch in range(100):
    train(model, train_loader)
    train_acc = test(model,train_loader)
    val_acc = test(model, val_loader)
    test_acc = test(model, test_loader)

    print('Epoch {}: TrainMAE {:.3f} ValMAE {:.3f} TestMAE {:.3f}'.format(epoch, train_acc, val_acc, test_acc))