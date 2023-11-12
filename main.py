import pdb
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork, Amazon

from config import cfg
from model.node_classification.mlp import MLPNet
from model.node_classification.gcn import GCNNet
from model.node_classification.gat import GATNet
from model.node_classification.gin import GINNet
from model.node_classification.sage import SAGE
from utils import set_global_seed, generate_mask, set_train_val_test_split


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Main entry")
    parser.add_argument('--cfg', dest='cfg_file', default='configs/base.yaml',
                        help='Config file path', type=str)

    if len(sys.argv) == 1:
        print('Now you are using the default configs.')
        parser.print_help()

    return parser.parse_args()


def main(data, model, train_mask, val_mask, test_mask):

    data = data.to(device)

    optimizer = Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd)
    loss = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_val_test_acc = 0
    with tqdm(range(cfg.optim.epochs)) as tq:
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
                'Train': '{:.3}'.format(train_acc.item()),
                # 'ValLoss': '{:.3}'.format(val_loss.item()),
                'Val': '{:.3}'.format(val_acc.item()),
                'Test': '{:.3}'.format(test_acc.item())
            }
            tq.set_postfix(infos)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_test_acc = test_acc
    return best_val_test_acc 


if __name__ == '__main__':
    
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)    

    model_mapping = {
        'mlp': MLPNet,
        'gcn': GCNNet,
        'gat': GATNet,
        'gin': GINNet,
        'sage': SAGE
    }

    pdb.set_trace()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cfg.dataset.name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='../datasets', name=cfg.dataset.name, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif cfg.dataset.name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='../datasets', name=cfg.dataset.name)
        data = dataset[0]
    elif cfg.dataset.name in ['squirrel', 'chameleon']:
        dataset = WikipediaNetwork(root='../datasets', name=cfg.dataset.name)
        data = dataset[0]
    elif cfg.dataset.name in ['photo', 'computers']:
        dataset = Amazon(root='../datasets', name=cfg.dataset.name)
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask = generate_mask(
            data.num_nodes, 
            train_ratio=0.1, 
            val_ratio=0.2)
        

    
    final_test_acc = []
    for i in range(cfg.repeat):
        set_global_seed(cfg.seed + i * 1450)
        model = model_mapping[cfg.model.name](data.x.size(dim=1), 
                cfg.model.hidden_dim, data.y.max().item() + 1, 
                cfg.model.dropout).to(device)
        if len(data.train_mask.shape) > 1:
            for msk in range(data.train_mask.shape[-1]):
                train_mask = data.train_mask[:, msk]
                val_mask = data.val_mask[:, msk]
                test_mask = data.test_mask[:, msk]
                best_val_test_acc = main(data, model, train_mask, val_mask, test_mask)
                final_test_acc.append(best_val_test_acc.item() * 100)
        else:
            data = set_train_val_test_split(seed=42, data=data)
            # pdb.set_trace()
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            best_val_test_acc = main(data, model, train_mask, val_mask, test_mask)
            final_test_acc.append(best_val_test_acc.item() * 100)

    print('Mean: {:.2f}, Std: {:.2f}'.format(np.mean(final_test_acc) , np.std(final_test_acc)))


