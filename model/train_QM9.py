import argparse
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--device', type = int, default = 0)
parser.add_argument('--root', 'data/QM9/')
parser.add_argument('--embedding_dim', type = int, default = 64)
parser.add_argument('--convolution_dim', type = int, default = 300)
parser.add_argument('--dense_dim', type = int, default = 600)
parser.add_argument('--reduction_type', type = str, default = 'mean')
parser.add_argument('--learning_rate', type = int, default = 0.005)
parser.add_argument('--weight_decay', type = str, default = 8E-4)
parser.add_argument('--max_decay', type = int, default = 0.5)
parser.add_argument('--target_index', type = int, default = 4)
parser.add_argument('--t_start', type = int, default = monotonic())
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--best_validation_score', type = int, default = torch.inf)
parser.add_argument('--epoch0', type = int, default = 0)
parser.add_argument('--hidden_channels', type = int, default = 128)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--val_acc_history', type = list, default = [])
parser.add_argument('--all_acc_history', type = list, default = [])
args = parser.parse_args()
print(args)


dataset = WebKB(root=root, name=name, transform=NormalizeFeatures())
data = dataset[0]

split_idx = dataset.get_idx_split()
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]


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


print("\ttrain mse:", compute_accuracy(train_loader))
print("\tvalid mse:", compute_accuracy(valid_loader))
print("\ttest mse:", compute_accuracy(test_loader))
