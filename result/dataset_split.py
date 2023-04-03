import torch
import ptens
from Transforms import ToPtens_Batch
from typing import Callable
from torch_geometric.datasets import Planetoid, WebKB, Amazon, QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures, BaseTransform
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
print(dataset)
data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'train_rest',
                                  num_val = 433,
                                  num_test = 540)
data = transform_nodes(data)
print(data)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print("===============================================================================")
dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
print(dataset)
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 532,
                                  num_test = 665)
data = transform_nodes(data)
print(data)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print("===============================================================================")
dataset = WebKB(root='data/WebKB', name='Cornell', transform=NormalizeFeatures())
print(dataset)
data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 29,
                                  num_test = 36)
data = transform_nodes(data)
print(data)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print("===============================================================================")
dataset = Amazon(root='data/Amazon', name='Photo', transform=NormalizeFeatures())
print(dataset)
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 1224,
                                  num_test = 1530) 
data = transform_nodes(data)
print(data)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print("===============================================================================")
dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
print(dataset)
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 3154,
                                  num_test = 3943)
data = transform_nodes(data)
print(data)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print("===============================================================================")
dataset = WebKB(root='data/WebKB', name='Wisconsin', transform=NormalizeFeatures())
print(dataset)
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 40,
                                  num_test = 50)
data = transform_nodes(data)
print(data)
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print("===============================================================================")
class CleanupData(BaseTransform):
    def __call__(self, data):
        data.x = data.x
        data.y = data.y.float()
        return data
class ToPtens_Batch(BaseTransform):
    def __call__(self, data):
        data.G = ptens.graph.from_matrix(torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1),dtype=torch.float32),
                                                             size=(data.num_nodes,data.num_nodes)).to_dense())
        return data

on_learn_transform = ToPtens_Batch()
on_process_transform = CleanupData()
dataset = QM9(root = 'data/QM9/', pre_transform=on_process_transform)
print(dataset)
#print(dataset.shape)
# NMP; Cormorant
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [110831,10000,10000])
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, prefetch_factor=2)
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)
for i in train_loader:
    batch = on_learn_transform(i)
    print(batch.x, batch.x.shape,batch.G, batch.batch, batch.batch.shape)
    print("===============================================================================")
    




