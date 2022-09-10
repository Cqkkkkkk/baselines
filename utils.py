import torch
import numpy as np
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from typing import Union


def cal_edge_homophily(edge_index, labels):
    cnt = 0
    for u, v in edge_index.T:
        # u = u.item()
        # v = v.item()
        if labels[u] == labels[v]:
            cnt += 1
    return cnt / len(edge_index.T)


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def show_class_acc(pred, label):
    C = label.max() + 1
    for c in range(C):
        acc_c = (pred[label == c] == c).sum() / len((pred[label == c] == c))
        print('Class {}: Total {}, Acc {:.3}'.format(c, len((pred[label == c] == c)), acc_c))


def mask_heter_edges(edges, y, p=1, c=0):
    cnt = 0
    edge_mask = np.ones(edges.shape).astype(bool)
    for i, (u, v) in enumerate(edges.T):
        if y[u] == c and y[v] != c and np.random.rand() < p:
            edge_mask[0][i] = False
            edge_mask[1][i] = False
            cnt += 1
        if y[v] == c and y[u] != c and np.random.rand() < p:
            edge_mask[0][i] = False
            edge_mask[1][i] = False
            cnt += 1
    print('[Warning] Masking edges: {} edges masked'.format(cnt))
    edges = edges[edge_mask].reshape(2, -1)
    return edges


def generate_mask(num_nodes, train_ratio, val_ratio):
    print('[Warning] Using non-public split')
    test_ratio = 1 - train_ratio - val_ratio
    train_mask = np.full((num_nodes), False)
    val_mask = np.full((num_nodes), False)
    test_mask = np.full((num_nodes), False)

    permute = np.random.permutation(num_nodes)
    train_idx = permute[: int(train_ratio * num_nodes)]
    val_idx = permute[int(train_ratio * num_nodes): int((train_ratio + val_ratio) * num_nodes)]
    test_idx = permute[int(1 - test_ratio * num_nodes):]
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def add_self_loops_to_isolated(edge_index, num_nodes):
    edges, a, node_mask = remove_isolated_nodes(edge_index, num_nodes=num_nodes)
    nodes = [i for i, x in enumerate(~node_mask) if x]
    new_edges = edge_index.T

    for node in nodes:
        loop_edge = torch.LongTensor([node, node]).unsqueeze(dim=0)
        new_edges = torch.cat((new_edges, loop_edge))

    return new_edges.T


# Transform to convert feature from Long to Float
class Int2Float(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        data.node_stores[0]['x'] =  data.node_stores[0]['x'].float()
        return data

