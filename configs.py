import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset', type=str, default='cora')
arg_parser.add_argument('--used_mask',  type=int, default=0)
arg_parser.add_argument('--seed',  type=int, default=42)
arg_parser.add_argument('--hidden_dim', type=int, default=128)
arg_parser.add_argument('--epoch', type=int, default=200)
arg_parser.add_argument('--lr', type=float, default=1e-3)
arg_parser.add_argument('--wd', type=float, default=1e-5)
arg_parser.add_argument('--dp', type=float, default=0.5)
arg_parser.add_argument('--repeat', type=int, default=10)
arg_parser.add_argument('--model', type=str, default='gat', choices=['mlp', 'gcn', 'gat', 'gin'])
args = arg_parser.parse_args()
