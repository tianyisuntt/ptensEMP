import torch
import ptens
from Transforms import ToPtens_Batch
from typing import Callable
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'train_rest',
                                  num_val = 433,
                                  num_test = 540)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Number of validation nodes: {data.val_mask.sum()}')
print(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
print(f'Number of testing nodes: {data.test_mask.sum()}')
print(f'Testing node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


class ConvolutionalLayer(torch.nn.Module):
    def __init__(self,channels_in: int,channels_out: int, nhops: int) -> None:
        super().__init__()
        self.nhops = nhops
        self.lin1 = ptens.modules.Linear(channels_in * 2,channels_in)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)
        self.lin2 = ptens.modules.Linear(channels_in, channels_out)
    def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
        x1 = x.transfer1(G.nhoods(self.nhops), G, False)
        x1 = self.lin1(x1).relu()
        x1 = self.dropout(x1)
        x1 = self.lin2(x1)
        return x1

class Model(torch.nn.Module):
    def __init__(self, channels_in: int, convolution_dim: int, dense_dim: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvolutionalLayer(channels_in,   convolution_dim, 1)
        self.conv2 = ConvolutionalLayer(convolution_dim, dense_dim,     4)
        self.lin1 = ptens.modules.Linear(dense_dim,dense_dim)
        self.drop = ptens.modules.Dropout(prob=0.5,device = None)
        self.lin2 = ptens.modules.Linear(dense_dim,out_channels)
    def forward(self, x: torch.Tensor, G: ptens.graph, batch: torch.Tensor) -> torch.Tensor:
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x)
        x = self.conv1(x,G)
        x = self.conv2(x,G) 
        x = self.lin1(x).relu()
        x = self.drop(x)
        x = self.lin2(x)
        x = ptens.linmaps0(x,False)
        return x



def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x,data.G, None)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step() 
      return loss
def test():
      model.eval()
      out = model(data.x,data.G, None)
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
      val_correct = pred[data.val_mask] == data.y[data.val_mask]  
      val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, val_acc, test_acc

channels_in = dataset.num_features
out_channels = dataset.num_classes
convolution_dim = 64
dense_dim = 128
reduction_type = 'mean'
model = Model(channels_in, convolution_dim,dense_dim,out_channels)
lr = 0.01
wd = 8e-1
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}', ", Train Accuracy:", train_acc, ", Validation Accuracy", val_acc, ", Test Accuracy:", test_acc, ".")
print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy:', train_acc,", Validation Accuracy:", val_acc, ", Test Accuracy:", test_acc, ".")
print('=================================================================')



