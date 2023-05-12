import os
import torch
import ptens
from typing import Callable
from time import monotonic
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

class CleanupData(BaseTransform):
    def __call__(self, data):
        data.x = data.x
        data.y = data.y.float()
        return data
class ToPtens_Batch(BaseTransform):
    def __call__(self, data):
        data.G = ptens.graph.from_matrix(torch.sparse_coo_tensor(
            data.edge_index,torch.ones(data.edge_index.size(1),dtype=torch.float32),
            size=(data.num_nodes,data.num_nodes)).to_dense())
        return data

on_process_transform = CleanupData()
on_learn_transform = ToPtens_Batch()

dataset = QM9(root = 'data/QM9/', pre_transform=on_process_transform)
print('dataset', dataset)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [110831,10000,10000])
print(len(train_set), len(val_set), len(test_set))
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, prefetch_factor=2)
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)

