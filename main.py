import time
import torch
import numpy as np
import torch.nn as nn
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, NELL

from model.mlp import MLPNet
from model.gcn import GCNNet
from model.gat import GATNet
from model.gin import GINNet

from utils import set_global_seed, show_class_acc, mask_heter_edges
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
    data.train_mask = data.train_mask[:, args.used_mask]
    data.val_mask = data.val_mask[:, args.used_mask]
    data.test_mask = data.test_mask[:, args.used_mask]
elif args.dataset in ['squirrel', 'chameleon']:
    dataset = WikipediaNetwork(root='./data', name=args.dataset)
    data = dataset[0]
    data.train_mask = data.train_mask[:, args.used_mask]
    data.val_mask = data.val_mask[:, args.used_mask]
    data.test_mask = data.test_mask[:, args.used_mask]
elif args.dataset in ['NELL']:
    dataset = NELL(root='./data')
    data = dataset[0]
    

if args.cal_degree:
    adj = to_dense_adj(data.edge_index.cpu()).squeeze(dim=0)
    node_degree = adj.sum(dim=0) + adj.sum(dim=1)
    node_degree = node_degree.numpy().astype(int)

for i in range(args.repeat):
    set_global_seed(args.seed + i * 10)
    model = Net(data.x.size(dim=1), args.hidden_dim, data.y.max().item() + 1, args.dp).to(device)
    data = data.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss = nn.CrossEntropyLoss()

    best_acc = 0
    best_loss = 9999999
    time_start = time.time()
    with tqdm(range(args.epoch)) as tq:
        for epoch in tq:
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss_t = loss(output[data.train_mask], data.y[data.train_mask])
            loss_t.backward()
            optimizer.step()
            train_loss = loss_t.detach()
            pred = output.argmax(dim=1)
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
            model.eval()
            with torch.no_grad():
                output = model(data)
                loss_t = loss(output[data.val_mask], data.y[data.val_mask])
                val_loss = loss_t
                pred = output.argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
            infos = {
                'Epoch': epoch,
                'TrainLoss': '{:.3}'.format(train_loss.item()),
                'TrainAcc': '{:.3}'.format(train_acc.item()),
                'ValLoss': '{:.3}'.format(val_loss.item()),
                'ValAcc': '{:.3}'.format(val_acc.item())
            }
            tq.set_postfix(infos)
            if val_acc > best_acc:
                best_loss = val_loss
                best_acc = val_acc
                torch.save(model.state_dict(), './ckpt/{}.pt'.format(args.model))
                # print('Saved')
    time_end = time.time()
    time_spent = time_end - time_start
    print('Time spent: {:.3}, {:.3} iter per sec'.format(time_spent, args.epoch / time_spent))       
    model = Net(data.x.size(dim=1), args.hidden_dim, data.y.max().item() + 1, args.dp).to(device)
    model.load_state_dict(torch.load('./ckpt/{}.pt'.format(args.model)))
    model.eval()
    output = model(data)
    
    loss_t = loss(output[data.test_mask], data.y[data.test_mask])
    test_loss = loss_t
    pred = output.argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    print('TestLoss: {:.3}, TestAcc: {:.3}'.format(test_loss.item(), test_acc.item() * 100))
    # show_class_acc(pred[data.test_mask], data.y[data.test_mask])

