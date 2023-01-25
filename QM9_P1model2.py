import os
import torch
import ptens
from typing import Callable
from time import monotonic
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
from torch_geometric.transforms import BaseTransform, Compose, NormalizeFeatures
from Transforms import PreComputeNorm, ToPtens_Batch

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

class ConvolutionalLayer(torch.nn.Module):
    def __init__(self, channels_in: int, channels_out: int, nhops: int) -> None:
        super().__init__()
        self.nhops = nhops
        hidden_dim = 128
        self.lin1 = ptens.modules.Linear(channels_in*2, hidden_dim)
        self.batchnorm1 = ptens.modules.BatchNorm(hidden_dim) 
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)
        self.lin2 = ptens.modules.Linear(hidden_dim, channels_out)
        
    def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
        x1 = x.transfer1(G.nhoods(self.nhops),G,False)
        atoms = x1.get_atoms()
        x = x1.torch()
        x2 = ptens.ptensors1.from_matrix(x,atoms)
        x2 = self.lin1(x1).relu()
        x2 = self.batchnorm1(x2)
        x2 = self.dropout(x2)
        x2 = self.lin2(x2)
        return x2
    
class Model(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dense_dim: int) -> None:
        super().__init__()
        self.embedding = ptens.modules.Linear(11,embedding_dim)
        self.conv1 = ConvolutionalLayer(embedding_dim, hidden_dim, 1)
        self.conv2 = ConvolutionalLayer(hidden_dim, hidden_dim,    2)
        self.conv3 = ConvolutionalLayer(hidden_dim, hidden_dim,    3)
        self.conv4 = ConvolutionalLayer(hidden_dim, dense_dim,     4)
        self.batchnorm = ptens.modules.BatchNorm(dense_dim) 
        self.lin1 = ptens.modules.Linear(dense_dim,dense_dim)
        self.lin2 = ptens.modules.Linear(dense_dim,dense_dim)
        self.lin3 = ptens.modules.Linear(dense_dim,dense_dim)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)
    def forward(self, x: torch.Tensor, G: ptens.graph, batch: torch.Tensor) -> torch.Tensor:
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x, False)
        x = self.embedding(x)
        x = self.conv1(x,G)
        x = self.conv2(x,G)
        x = self.conv3(x,G)
        x = self.conv4(x,G)
        x = self.batchnorm(x)
        x = self.lin1(x).relu()
        x = self.dropout(x)
        x = self.lin2(x).relu()
        x = self.dropout(x)
        x = self.lin3(x)
        x = ptens.linmaps0(x, False)
        x = x.torch()
        return x


embedding_dim = 64
convolution_dim = 300
dense_dim = 600
reduction_type = 'mean'
model = Model(embedding_dim,convolution_dim,dense_dim)
print(model)


learning_rate = 0.00005
weight_decay = 8E-4
optimizer = torch.optim.Adam(model.parameters(),
                             learning_rate,
                             weight_decay=weight_decay)
criterion = torch.nn.MSELoss()
max_decay = 0.5
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   verbose=True,
                                                   factor=max_decay,
                                                   mode = 'min',
                                                   patience=4)


target_index = 0
def compute_accuracy(loader):
    global target_index
    model.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for subset in loader:
            batch = on_learn_transform(subset)
            pred = model(batch.x,batch.G,batch.batch)[:,target_index]
            preds.append(pred)
            targets.append(batch.y[:,target_index])
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    model.train()
    mse =torch.nn.functional.mse_loss(preds,targets)
    l1 = torch.nn.functional.l1_loss(preds,targets)
    return [mse, l1]

T = 0
t_start = monotonic()
epochs = 20# 2#200
val_acc_history = []
all_acc_history = []
best_validation_score = torch.inf
epoch0 = 0
best_validation_scores = []
for epoch in range(epoch0 + 1,epochs):
    total = 0
    index = 0
    t0 = monotonic()
    for subset in train_loader:
        batch = on_learn_transform(subset)
        optimizer.zero_grad()
        loss = criterion(model(batch.x,batch.G,batch.batch)[:,target_index],batch.y[:,target_index])
        loss.backward()
        optimizer.step()
        inc = loss.tolist()
        index += 1
        total += inc
        print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, loss: %7.5f" \
              % (epoch,index,len(train_loader),total/index,inc),end='',flush=True)

    t1 = monotonic()
    dt = t1 - t0
    T += dt
    avg_t_per_epoch = (t1 - t_start)/(epoch - epoch0 + 2)
    val_acc = compute_accuracy(valid_loader)
    val_acc_history.append(val_acc)
    print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, average time/epoch: %7.3f (s), time for epoch: %7.3f (s), estimated time remaining: %7.3f (s), valid scores: %5.3f" \
          % (epoch,index,len(train_loader),total/index,avg_t_per_epoch,dt,(epochs - epoch - 1)*avg_t_per_epoch,val_acc))

    train_acc = compute_accuracy(train_loader)
    test_acc = compute_accuracy(test_loader)
    print("train mse; l1",train_acc)
    print("valid mse; l1",val_acc)
    print("test  mse; l1",test_acc)
    all_acc_history.append((train_acc,val_acc,test_acc))
    model.eval()
    if val_acc[1] < best_validation_score:
        best_validation_scores = val_acc
        print("New best validation scores [mse, l1]:",best_validation_scores),
    model.train()
    sched.step(val_acc)
    print() 

print("\ttrain mse; l1",compute_accuracy(train_loader))
print("\tvalid mse; l1",compute_accuracy(valid_loader))
print("\ttest  mse; l1",compute_accuracy(test_loader))

