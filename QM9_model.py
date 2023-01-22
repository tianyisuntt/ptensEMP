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
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [100000, 10000, 20831])
print(len(train_set), len(val_set), len(test_set))
batch_size = 256
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
        x = self.lin2(x)
        return x


embedding_dim = 64
convolution_dim = 300
dense_dim = 600
reduction_type = 'mean'
model = Model(embedding_dim,convolution_dim,dense_dim,global_mean_pool)
print(model)


learning_rate = 0.005
weight_decay = 1E-4
max_decay = 0.5
optimizer = torch.optim.Adam(model.parameters(),
                             learning_rate,
                             weight_decay=weight_decay)
criterion = torch.nn.MSELoss()
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
    return torch.nn.functional.mse_loss(preds,targets)

T = 0
t_start = monotonic()
epochs = 20#200
val_acc_history = []
all_acc_history = []
best_validation_score = 0
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
        if index % 10 == 0:
            print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, loss: %7.5f" \
                  % (epoch,index,len(train_loader),total/index,inc),end='',flush=True)

    t1 = monotonic()
    dt = t1 - t0
    T += dt
    avg_t_per_epoch = (t1 - t_start)/(epoch - epoch0 + 2)
    val_acc = compute_accuracy(valid_loader)
    val_acc_history.append(val_acc)
    print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, average time/epoch: %7.3f (s), time for epoch: %7.3f (s), estimated time remaining: %7.3f (s), valid score: %5.3f" \
          % (epoch,index,len(train_loader),total/index,avg_t_per_epoch,dt,(epochs - epoch - 1)*avg_t_per_epoch,val_acc))
    if (epoch % 10 == 0 or epoch + 1 == epochs):
        train_acc = compute_accuracy(train_loader)
        test_acc = compute_accuracy(test_loader)
        print("train mse:",train_acc)
        print("test  mse:",test_acc)
        print("valid mse:",val_acc)
        all_acc_history.append((train_acc,test_acc,val_acc))
    model.eval()
    if val_acc < best_validation_score:
        best_validation_score = val_acc
        print("New best validation score:",val_acc),
    model.train()
    sched.step(val_acc)
    print() 

print("\ttrain mse:",compute_accuracy(train_loader))
print("\ttest  mse:",compute_accuracy(test_loader))
print("\tvalid mse_loss:",compute_accuracy(valid_loader))
'''
train mse: tensor(71564.3828)
test  mse: tensor(484.0068)
valid mse: tensor(146342.2188)
'''
