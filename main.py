import torch
import torch.nn as nn
import numpy as np
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork

from model.mlp import MLPNet
from model.gcn import GCNNet
from model.gat import GATNet
from model.gin import GINNet

from utils import set_global_seed
from configs import args

print(args)


if args.model == 'mlp':
    Net = MLPNet
elif args.model == 'gcn':
    Net = GCNNet
elif args.model == 'gat':
    Net = GATNet
elif args.model == 'gin':
    Net = GINNet
else:
    raise NotImplementedError


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if args.dataset in ['cora', 'citeseer', 'pubmed']:
    dataset = Planetoid(root='./data', name=args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
elif args.dataset in ['texas', 'cornell', 'wisconsin']:
    dataset = WebKB(root='./data', name=args.dataset)
    data = dataset[0]
elif args.dataset in ['squirrel', 'chameleon']:
    dataset = WikipediaNetwork(root='./data', name=args.dataset)
    data = dataset[0]
    

for i in range(args.repeat):
    final_test_acc = []
    set_global_seed(args.seed + i * 10)
    for msk in range(data.train_mask.shape[1]):
        train_mask = data.train_mask[:, msk]
        val_mask = data.val_mask[:, msk]
        test_mask = data.test_mask[:, msk]
        
        model = Net(data.x.size(dim=1), args.hidden_dim, data.y.max().item() + 1, args.dp).to(device)
        data = data.to(device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        loss = nn.CrossEntropyLoss()

        best_epoch = 0
        best_val_acc = 0
        best_val_test_acc = 0
        best_val_loss = 9999999
        with tqdm(range(args.epoch)) as tq:
            for epoch in tq:
                model.train()
                optimizer.zero_grad()
                output = model(data)
                loss_t = loss(output[train_mask], data.y[train_mask])
                loss_t.backward()
                optimizer.step()
                train_loss = loss_t.detach()
                pred = output.argmax(dim=1)
                train_acc = (pred[train_mask] == data.y[train_mask]).sum() / train_mask.sum()
                model.eval()
                with torch.no_grad():
                    output = model(data)
                    loss_t = loss(output[val_mask], data.y[val_mask])
                    val_loss = loss_t
                    pred = output.argmax(dim=1)
                    val_acc = (pred[val_mask] == data.y[val_mask]).sum() / val_mask.sum()
                    test_acc = (pred[test_mask] == data.y[test_mask]).sum() / test_mask.sum()
                infos = {
                    'Epoch': epoch,
                    'TrainLoss': '{:.3}'.format(train_loss.item()),
                    'TrainAcc': '{:.3}'.format(train_acc.item()),
                    'ValLoss': '{:.3}'.format(val_loss.item()),
                    'ValAcc': '{:.3}'.format(val_acc.item()),
                    'TestAcc': '{:.3}'.format(test_acc.item())
                }
                tq.set_postfix(infos)
                if val_acc > best_val_acc:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_val_test_acc = test_acc
                    best_epoch = epoch
            final_test_acc.append(best_val_test_acc.item() * 100)
        
    print('Mean: {:.3f}, Std: {:.3f}'.format(np.mean(final_test_acc) , np.std(final_test_acc)))


