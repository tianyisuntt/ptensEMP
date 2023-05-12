import torch
import ptens
from typing import Callable
from time import monotonic
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
from torch_geometric.transforms import BaseTransform, Compose, NormalizeFeatures
from Trasnforms import PreComputeNorm, ToPtens_Batch


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

target_index = 2
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
