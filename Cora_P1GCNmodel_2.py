import torch
import ptens
from Transforms import ToPtens_Batch
from typing import Callable
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 500,
                                  num_val = 300)

data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)
class ConvolutionalLayer(torch.nn.Module):
    def __init__(self,channels_in: int,channels_out: int, nhops: int) -> None:
        super().__init__()
       # self.batchnorm1 = torch.nn.BatchNorm1d(channels_in * 2)
        self.lin1 = torch.nn.Linear(channels_in * 2,channels_in)
        self.activ1 = torch.nn.ReLU(True)
       # self.batchnorm2 = torch.nn.BatchNorm1d(channels_in)
        self.lin2 = torch.nn.Linear(channels_in, channels_out)
        self.activ2 = torch.nn.ReLU(True)
        self.nhops = nhops
    def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
        x1 = x.transfer1(G.nhoods(self.nhops), G, True)
        atoms = x1.get_atoms()
        x2 = x1.torch()
      #  x2 = self.batchnorm1(x2)
        x2 = self.activ1(self.lin1(x2))
      #  x2 = self.batchnorm2(x2)
        x2 = self.activ2(self.lin2(x2))
        x3 = ptens.ptensors1.from_matrix(x2,atoms)
        return x3

class Model(torch.nn.Module):
    def __init__(self, embedding_dim: int, convolution_dim: int, dense_dim: int,
                 pooling: Callable[[torch.Tensor,torch.Tensor],torch.Tensor]) -> None:
        super().__init__()
        self.embedding = torch.nn.Linear(1433,embedding_dim)
        self.conv1 = ConvolutionalLayer(embedding_dim,   convolution_dim, 1)
        self.conv2 = ConvolutionalLayer(convolution_dim, convolution_dim, 2)
        self.conv3 = ConvolutionalLayer(convolution_dim, convolution_dim, 3)
        self.conv4 = ConvolutionalLayer(convolution_dim, dense_dim,       4)
        self.pooling = pooling
       # self.batchnorm = torch.nn.BatchNorm1d(dense_dim)
        self.lin1 = torch.nn.Linear(dense_dim,dense_dim)
        self.activ1 = torch.nn.ReLU(True)
        self.lin2 = torch.nn.Linear(dense_dim,dense_dim)
        self.activ2 = torch.nn.ReLU(True)
        self.lin3 = torch.nn.Linear(dense_dim,dense_dim)
    def forward(self, x: torch.Tensor, G: ptens.graph, batch: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x)
        x = self.conv1(x,G)
        x = self.conv2(x,G)
        x = self.conv3(x,G)
        x = self.conv4(x,G)
        x = ptens.linmaps0(x,True)
        x = x.torch()
        #x = self.pooling(x,batch)
      #  x = self.batchnorm(x)
        x = self.activ1(self.lin1(x))
        x = self.activ2(self.lin2(x))
        x = self.lin2(x)
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
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, test_acc


embedding_dim = 64
convolution_dim = 300
dense_dim = 600
reduction_type = 'mean'
model = Model(embedding_dim,convolution_dim,dense_dim,global_mean_pool) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
'''
Killed: 9
'''


