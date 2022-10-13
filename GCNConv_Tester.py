import torch
from GCNConv import GCNConv
from torch_geometric.nn import GCNConv as PyG_GCNConv
from torch_geometric.nn import Sequential
from torch_geometric.datasets import Planetoid
from torch.nn import EmbeddingBag, ReLU, Linear, Softmax
import ptens
# hyper parameters
use_pyg_model = False
learning_rate = 0.001
decay = 0.5
epochs = 20
warmup_epochs = 10
# getting dataset
dataset = Planetoid('datasets','CORA')
graph = dataset[0]
train_mask, val_mask, test_mask = graph.train_mask, graph.val_mask, graph.test_mask
x = graph.x
y = graph.y
in_channels = graph.x.size(1)
out_channels = graph.y.size(1)
# defining model
if use_pyg_model:
  G = graph.edge_index
  model = Sequential('x,G',[
    (EmbeddingBag(in_channels,128),'x->x'),
    ReLU(True),
    (PyG_GCNConv(128,128,add_self_loops=False,normalize=False),'x,G->x'),
    ReLU(True),
    (GCNConv(128,128),'x,G->x'),
    Linear(128,out_channels),
    Softmax(1)
  ])
else:
  G = ptens.graph.from_matrix(graph.coo())
  model = Sequential('x,G',[
    (EmbeddingBag(in_channels,128),'x->x'),
    ReLU(True),
    (GCNConv(128,128),'x,G->x'),
    ReLU(True),
    (GCNConv(128,128),'x,G->x'),
    Linear(128,out_channels),
    Softmax(1)
  ])
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
  loss(model(x,G)[train_mask],train_y).backward()
  optimizer.step()
  if epoch >= warmup_epochs:
    schedular.step()
