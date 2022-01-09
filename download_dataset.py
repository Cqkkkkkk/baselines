from torch_geometric.datasets import Amazon

dataset = Amazon(root='./data', name='Computers')

data = dataset[0]

print(data)