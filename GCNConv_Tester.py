from turtle import forward
import torch
from ptens_layers import GCNConv, Linear
#from torch_geometric.nn import GCNConv as PyG_GCNConv, Sequential
from torch_geometric.nn import GCNConv as PyG_GCNConv
#from torch_geometric.nn import Sequential
import torch_geometric.nn as nn
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import Planetoid
from torch.nn import EmbeddingBag, ReLU, Softmax
import ptens
# hyper parameters
use_pyg_model = False
learning_rate = 0.001
decay = 0.5
epochs = 20
warmup_epochs = 10
# getting dataset
class ToInt(BaseTransform):
  def __call__(self, data):
    data.x = data.x.type(torch.int32)
    return data
dataset = Planetoid('datasets','CORA',pre_transform=ToInt())
in_channels = dataset.num_features
out_channels = dataset.num_classes
graph = dataset[0]
train_mask, val_mask, test_mask = graph.train_mask, graph.val_mask, graph.test_mask
x = graph.x
y = graph.y
G = ptens.graph.from_matrix(torch.sparse_coo_tensor(graph.edge_index,torch.ones(graph.edge_index.size(1),dtype=bool)).float().to_dense())

#
# defining model
def print_type(x):
  print('type(x) = ',type(x))
  print('x.dtype = ',x.dtype)
  print('x.size() = ',x.size())
  print("~~",type(ptens.ptensors0.from_matrix(x)))
  return x
"""
model = nn.Sequential('x,G',[
  (EmbeddingBag(in_channels,128),'x -> x'),
  #from_matrix,
  lambda a: ptens.ptensors0.from_matrix(a),
  #print_type(x),
  ptens.relu,
  #print_type(x),
  (GCNConv(128,128),'x,G -> x'),
  ptens.relu,
  (GCNConv(128,128),'x,G -> x'),
  Linear(128,out_channels),
  lambda x: x.torch(),
  Softmax(1)
])
"""
class Model(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.embedding = EmbeddingBag(in_channels,128)
    self.conv1 = GCNConv(128,128,False)
    self.conv2 = GCNConv(128,128,False)
    self.lin = Linear(128,out_channels,False)
  def forward(self, x: torch.Tensor, G: ptens.graph) -> torch.Tensor:
    x = self.embedding(x)
    x = ptens.ptensors0.from_matrix(x)
    x = ptens.relu(x)
    x = self.conv1(x,G)
    x = ptens.relu(x)
    x = self.conv2(x,G)
    x = self.lin(x)
    x = x.torch()
    #x = x.sum()
    #x.backward()
    x = torch.softmax(x,1)
    return x
model = Model()

# running tests
def compute_accuracy(mask: torch.Tensor) -> float:
  with torch.no_grad():
    predictions = model(x,G)[mask].detach().argmax(1)
  labels = y[mask].detach()
  return (predictions == labels).float().mean()

loss = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer,decay)
train_y = y[train_mask]
for epoch in range(epochs):
  optimizer.zero_grad()
  l = loss(model(x,G)[train_mask],train_y)
  print("loss:",l)
  l.backward()
  optimizer.step()
  if epoch >= warmup_epochs:
    schedular.step()
