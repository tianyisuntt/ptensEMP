from typing import Callable
from torch_geometric.loader import DataLoader
from time import monotonic
import torch
import ptens
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
from torch_geometric.transforms import BaseTransform, Compose
from Transforms import PreComputeNorm, ToPtens_Batch
import os
from torch_geometric.datasets import QM9

# Hyperparameters
dataset_name = 'qm9'
learning_rate = 0.005
batch_size = 32*4#32*2
embedding_dim = 128
convolution_dim = 300
dense_dim = 600
epochs = 200
weight_decay = 1E-4
max_decay = 0.5
target_index = 0
val_history = []
history = []

best_validation_model_path = 'qm9_val_model.pt'
model_path = 'qm9_model.pt'
reduction_type = 'mean'
try_reload = True

##
best_validation_score = 0

class CleanupData(BaseTransform):
  def __call__(self, data):
    data.x = data.x.to('cuda')
    data.y = data.y.float().cuda()
    return data
class ToPtens_Batch(BaseTransform):
  def __call__(self, data):
    data.G = ptens.graph.from_matrix(torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1),dtype=torch.float32),
                                                             size=(data.num_nodes,data.num_nodes)).to_dense())
    return data

on_process_transform = CleanupData()
on_learn_transform = ToPtens_Batch()

dataset = QM9(root = 'data/QM9/', pre_transform=on_process_transform)
print('dataset', dataset)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [100000, 10000, 20831])
print(len(train_set), len(val_set), len(test_set))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, prefetch_factor=2)
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, prefetch_factor=2)

class ConvolutionalLayer(torch.nn.Module):
  def __init__(self,channels_in: int,channels_out: int,nhops: int) -> None:
    super().__init__()
    self.batchnorm1 = torch.nn.BatchNorm1d(channels_in * 2)
    self.lin1 = torch.nn.Linear(channels_in * 2,channels_in)
    self.activ1 = torch.nn.ReLU(True)
    self.batchnorm2 = torch.nn.BatchNorm1d(channels_in )
    self.lin2 = torch.nn.Linear(channels_in ,channels_out)
    self.activ2 = torch.nn.ReLU(True)
    self.nhops = nhops
  def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
    x1 = x.transfer1(G.nhoods(self.nhops),G,True)
    atoms = x1.get_atoms()
    x2 = x1.torch()
    #
    x2 = self.batchnorm1(x2)
    x2 = self.activ1(self.lin1(x2))
    #
    x2 = self.batchnorm2(x2)
    x2 = self.activ2(self.lin2(x2))
    x3 = ptens.ptensors1.from_matrix(x2,atoms)
    return x3

class Model(torch.nn.Module):
  def __init__(self, embedding_dim: int, convolution_dim: int, dense_dim: int,
               pooling: Callable[[torch.Tensor,torch.Tensor],torch.Tensor]) -> None:
    super().__init__()
    self.embedding = torch.nn.Linear(11,embedding_dim)
    self.conv1 = ConvolutionalLayer(embedding_dim,   convolution_dim, 1)
    self.conv2 = ConvolutionalLayer(convolution_dim, convolution_dim, 2)
    self.conv3 = ConvolutionalLayer(convolution_dim, convolution_dim, 3)
    self.conv4 = ConvolutionalLayer(convolution_dim, dense_dim,       4)
    self.pooling = pooling
    self.batchnorm = torch.nn.BatchNorm1d(dense_dim)
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
    x = self.pooling(x,batch)
    x = self.batchnorm(x)
    x = self.activ1(self.lin1(x))
    x = self.activ2(self.lin2(x))
    x = self.lin3(x)
    return x

model = Model(embedding_dim,convolution_dim,dense_dim,global_mean_pool)

try_reload = try_reload and os.path.exists(model_path)

if try_reload:
  checkpoint = torch.load(model_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch0 = checkpoint['epoch']
  best_validation_score = checkpoint['best_validation_score']
else:
  epoch0 = -1
model.cuda()

optimizer = torch.optim.Adam(model.parameters(),learning_rate,weight_decay=weight_decay)
loss = torch.nn.MSELoss()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,
                                                   factor=max_decay,mode = 'min',patience=4)


if try_reload:
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  sched.load_state_dict(checkpoint['sched'])
model.train()

def compute_accuracy(loader):
  global target_index
  model.eval()
  with torch.no_grad():
    predictions = []
    targets = []
    for batch in loader:
      batch = on_learn_transform(batch)
      pred = model(batch.x,batch.G,None,batch.batch.cuda())
      predictions.append(pred)
      targets.append(batch.y[:,target_index])
  predictions = torch.cat(predictions)
  targets = torch.cat(targets)
  model.train()
  return torch.nn.functional.mse_loss(predictions,targets)

T = 0
t_start = monotonic()
for epoch in range(epoch0 + 1,epochs):
  print('epoch:========', epoch)
  tot = 0
  ind = 0
  t0 = monotonic()
  for batch in train_loader:
    batch = on_learn_transform(batch)
    optimizer.zero_grad()
    l = loss(model(batch.x,batch.G,batch.batch.cuda())[:,target_index],batch.y[:,target_index])
    l.backward()
    optimizer.step()
    inc = l.tolist()
    ind += 1
    tot += inc
    if ind % 10 == 0:
      print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, loss: %7.5f" % (epoch,ind,len(train_loader),tot/ind,inc),end='',flush=True)
  torch.save({
    'epoch'                 : epoch                  ,
    'model_state_dict'      : model.state_dict()     ,
    'optimizer_state_dict'  : optimizer.state_dict() ,
    'best_validation_score' : best_validation_score  ,
    'sched'                 : sched.state_dict()     ,
  }, model_path)
  t1 = monotonic()
  dt = t1 - t0
  T += dt
  avg_t_per_epoch = (t1 - t_start)/(epoch - epoch0 + 2)
  # total_epochs = 1/ avg_t_per_epoch * (t_last - t1) + epoch + 1
  # => epochs_remaining = (t_last - t1)/avg_t_per_epoch
  # => epochs_remaining * avg_t_per_epoch = t_last - t1
  val = compute_accuracy(valid_loader)
  val_history.append(val)
  print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, average time/epoch: %7.3f (s), time for epoch: %7.3f (s), estimated time remaining: %7.3f (s), valid score: %5.3f" \
    % (epoch,ind,len(train_loader),tot/ind,avg_t_per_epoch,dt,(epochs - epoch - 1)*avg_t_per_epoch,val))
  if (epoch % 10 == 0 or epoch + 1 == epochs):
    train_accuracy = compute_accuracy(train_loader)
    test_accuracy = compute_accuracy(test_loader)
    print("train accuracy:",train_accuracy)
    print("test  accuracy:",test_accuracy)
    print("valid accuracy:",val)
    history.append((train_accuracy,test_accuracy,val))
  # checking validation score
  model.eval()
  if val < best_validation_score:
    best_validation_score = val
    print("New best validation score:",val),
    torch.save({
      'epoch'            : epoch               ,
      'model_state_dict' : model.state_dict()  ,
      'validation_score' : val                 ,
    }, best_validation_model_path)
  model.train()
  #
  sched.step(val)
  #sched.step()
  print()

print("Best validation results:")
checkpoint = torch.load(best_validation_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
val = checkpoint['validation_score']
print("\tEpoch:",epoch)
print("\ttrain accuracy:",compute_accuracy(train_loader))
print("\ttest  accuracy:",compute_accuracy(test_loader))
print("\tvalid accuracy:",val)
