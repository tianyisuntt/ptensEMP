import os
import torch
import ptens
from typinng import Callable
from time import monotonic
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
from torch_geometric.transforms import BaseTransform, Compose, NormalizeFeatures
from Transforms import PreComputeNorm, ToPtens_Batch
from QM9_model import *

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


embedding_dim = 64
convolution_dim = 300
dense_dim = 600
reduction_type = 'mean'
model = Model(embedding_dim,convolution_dim,dense_dim,global_mean_pool)
print(model)


learning_rate = 0.005
weight_decay = 8E-4
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


def test(target_index):  
    T = 0
    t_start = monotonic()
    epochs = 20
    val_acc_history = []
    all_acc_history = []
    best_validation_score = torch.inf
    epoch0 = 0
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
        print("\rEpoch %3d, batch: %5d/%5d, average loss: %7.5f, average time/epoch: %7.3f (s), time for epoch: %7.3f (s), estimated time remaining: %7.3f (s), valid score: %5.3f" \
              % (epoch,index,len(train_loader),total/index,avg_t_per_epoch,dt,(epochs - epoch - 1)*avg_t_per_epoch,val_acc))
        train_acc = compute_accuracy(train_loader)
        test_acc = compute_accuracy(test_loader)
        print("train mse:",train_acc)
        print("valid mse:",val_acc)
        print("test  mse:",test_acc)
        all_acc_history.append((train_acc,test_acc,val_acc))
        model.eval()
        if val_acc < best_validation_score:
            best_validation_score = val_acc
            print("New best validation score:",val_acc),
        model.train()
        sched.step(val_acc)
        print() 

    print("\ttrain mse:",compute_accuracy(train_loader))
    print("\tvalid mse:",compute_accuracy(valid_loader))
    print("\ttest  mse:",compute_accuracy(test_loader))



ls_target_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
for target_index in ls_target_index:
    test(target_index)











