from typing import List
import torch
import numpy as np
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
use_ptens_model = True 
learning_rate = 0.001
hidden_channels = 128
decay = 0.9
epochs = 100
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
G = ptens.graph.from_matrix(torch.sparse_coo_tensor(graph.edge_index,torch.ones(graph.edge_index.size(1),dtype=bool)).float().to_dense()) \
  if use_ptens_model else graph.edge_index

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
    self.embedding = EmbeddingBag(in_channels,hidden_channels)
    self.conv = [
      GCNConv(hidden_channels,hidden_channels,False) if use_ptens_model else PyG_GCNConv(hidden_channels,hidden_channels,normalize=False,bias=False)
      for i in range(5)
    ]
    self.lin = Linear(128,out_channels,False) if use_ptens_model else torch.nn.Linear(hidden_channels,out_channels,False)
    self.relu = ptens.relu if use_ptens_model else torch.nn.ReLU(True)
  def parameters(self, recurse: bool = True) -> List[torch.nn.Parameter]:
    params = []
    params += self.embedding.parameters()
    for c in self.conv:
      params += c.parameters()
    params += self.lin.parameters()
    return params
  def train(self, mode: bool = True):
    self.embedding.train(mode)
    [c.train(mode) for c in self.conv]
    self.lin.train(mode)
    return super().train(mode)
  def forward(self, x: torch.Tensor, G: ptens.graph) -> torch.Tensor:
    x = self.embedding(x)
    if use_ptens_model:
      x = ptens.ptensors0.from_matrix(x)
    x = self.relu(x)
    for c in self.conv:
      x = self.relu(c(x,G))
    x = self.lin(x)
    if use_ptens_model:
      x = x.torch()
    #x = torch.softmax(x,1)
    return x
model = Model()
print([s.size() for s in model.parameters()])

# running tests
def compute_accuracy(mask: torch.Tensor) -> float:
  with torch.no_grad():
    predictions = model(x,G)[mask].detach()
    #print(predictions)
    predictions = predictions.argmax(1)
  labels = y[mask].detach()
  #print(predictions)
  #print(labels)
  #raise Exception()
  return (predictions == labels).float().mean()

loss = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer,decay)
train_y = y[train_mask]
model.train()
init_weights = [p + 0 for p in model.parameters()]
val_history = []
train_history = []
test_history = []
loss_histroy = []
for epoch in range(epochs):
  print("\nepoch: %d" % (epoch + 1))
  optimizer.zero_grad()
  l = loss(model(x,G)[train_mask],train_y)
  loss_histroy.append(l.tolist())
  print("loss:",loss_histroy[-1])
  l.backward()
  #print(list(model.parameters())[-1].grad)
  optimizer.step()
  if epoch >= warmup_epochs:
    schedular.step()
  #weights = [p + 0 for p in model.parameters()]
  #same = [(init_weights[i] == weights[i]).all() for i in range(len(weights))]
  #print(np.argwhere(same))
  val_history.append(compute_accuracy(val_mask).tolist())
  train_history.append(compute_accuracy(train_mask).tolist())
  test_history.append(compute_accuracy(test_mask).tolist())
  print("val accuracy:",val_history[-1])
  print("train accuracy:",train_history[-1])
  print("test accuracy:",test_history[-1])
from matplotlib import pyplot as plt
plt.plot(train_history)
plt.plot(test_history)
plt.plot(val_history)
plt.legend(labels=['train','test','val'])
plt.savefig('accuracy_history.png')
plt.clf()
plt.plot(loss_histroy)
plt.savefig('loss_history.png')