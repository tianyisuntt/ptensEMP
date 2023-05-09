import torch
import ptens
from CiteSeer_models import *
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 510,
                                  num_val = 500)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)


def train():
      model.train()
      optimizer.zero_grad()
      out = model(ptens.ptensors0.from_matrix(data.x),data.G).torch()
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step() 
      return loss
def test():
      model.eval()
      out = model(ptens.ptensors0.from_matrix(data.x),data.G).torch()
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, test_acc


model = PMLP(hidden_channels = 256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
"""
CiteSeer:
num_train_per_class = 510, num_val = 500
hidden_channels = 256
epoches = 200
Train Accuracy: 
"""


model = P0GCN(hidden_channels = 32) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
"""
CiteSeer:
num_train_per_class = 510, num_val = 500
hidden_channels = 32
Train Accuracy: 0.7258179231863442 . Test Accuracy: 0.6666666666666666 .
"""


model = P1GCN0(hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
#"""
# Epoch: 200, Loss: 1.4102
# lr=0.001, weight_decay=8e-4
# Train Accuracy: 0.7098591549295775 . Test Accuracy: 0.6661654135338346 .


model = P1GCN2(embedding_dim = 64, hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
# Epoch: 022, Loss: 4.6239
# Train Accuracy: 0.21032863849765257 . Test Accuracy: 0.18045112781954886 .
# Epoch: 021, Loss: 5.0234
# Train Accuracy: 0.24553990610328638 . Test Accuracy: 0.2661654135338346 .


model = P1GCN(hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
# Epoch: 120, Loss: 1.4328
# Train Accuracy: 0.7136150234741784 . Test Accuracy: 0.6932330827067669 .
# Epoch: 126, Loss: 1.5311
# Train Accuracy: 0.7192488262910798 . Test Accuracy: 0.706766917293233 .
# Epoch: 128, Loss: 1.4973
# Train Accuracy: 0.7248826291079812 . Test Accuracy: 0.6992481203007519 .
# Epoch: 130, Loss: 1.3599
# Train Accuracy: 0.7192488262910798 . Test Accuracy: 0.6947368421052632 .
# Epoch: 133, Loss: 1.4939
# Train Accuracy: 0.7258215962441315 . Test Accuracy: 0.7022556390977444 .
# Epoch: 137, Loss: 1.4313
# Train Accuracy: 0.7267605633802817 . Test Accuracy: 0.6947368421052632 .
# Epoch: 145, Loss: 1.4055
# Train Accuracy: 0.7267605633802817 . Test Accuracy: 0.6992481203007519 .
# Epoch: 147, Loss: 1.3581
# Train Accuracy: 0.7370892018779343 . Test Accuracy: 0.7157894736842105 .
# Epoch: 148, Loss: 1.2950
# Train Accuracy: 0.7225352112676057 . Test Accuracy: 0.6977443609022557 .
# Epoch: 149, Loss: 1.3523
# Train Accuracy: 0.7173708920187793 . Test Accuracy: 0.6932330827067669 .
# Epoch: 150, Loss: 1.3162
# Train Accuracy: 0.7267605633802817 . Test Accuracy: 0.687218045112782 .
# Epoch: 151, Loss: 1.2487
# Train Accuracy: 0.7215962441314554 . Test Accuracy: 0.687218045112782 .
# Epoch: 153, Loss: 1.2745
# Train Accuracy: 0.7338028169014085 . Test Accuracy: 0.7037593984962406 .
# Epoch: 156, Loss: 1.3537
# Train Accuracy: 0.7267605633802817 . Test Accuracy: 0.6917293233082706 .
# Epoch: 157, Loss: 1.3577
# Train Accuracy: 0.7300469483568075 . Test Accuracy: 0.706766917293233 .
# Epoch: 159, Loss: 1.2987
# Train Accuracy: 0.7211267605633803 . Test Accuracy: 0.6827067669172933 .
# Epoch: 165, Loss: 1.7620
# Train Accuracy: 0.7272300469483568 . Test Accuracy: 0.6917293233082706 .
# Epoch: 166, Loss: 1.4541
# Train Accuracy: 0.7281690140845071 . Test Accuracy: 0.6857142857142857 .
# Epoch: 167, Loss: 1.2534
# Train Accuracy: 0.7309859154929578 . Test Accuracy: 0.7022556390977444 .
# Epoch: 168, Loss: 1.4525
# Train Accuracy: 0.7211267605633803 . Test Accuracy: 0.7007518796992481 .
# Epoch: 175, Loss: 1.4953
# Train Accuracy: 0.7286384976525822 . Test Accuracy: 0.6902255639097744 .
# Epoch: 184, Loss: 1.6953
# Train Accuracy: 0.6995305164319249 . Test Accuracy: 0.6887218045112782 .
# Epoch: 187, Loss: 1.4312
# Train Accuracy: 0.7206572769953051 . Test Accuracy: 0.687218045112782 .
# Epoch: 189, Loss: 1.4639
# Train Accuracy: 0.7319248826291079 . Test Accuracy: 0.6932330827067669 .
# Epoch: 190, Loss: 1.4417
# Train Accuracy: 0.7206572769953051 . Test Accuracy: 0.6932330827067669 .
# Epoch: 191, Loss: 1.4921
# Train Accuracy: 0.7380281690140845 . Test Accuracy: 0.6977443609022557 .
# Epoch: 192, Loss: 1.4157
# Train Accuracy: 0.7215962441314554 . Test Accuracy: 0.6887218045112782 .
# Epoch: 193, Loss: 1.2742
# Train Accuracy: 0.7131455399061033 . Test Accuracy: 0.681203007518797 .
# Epoch: 195, Loss: 1.3770
# Train Accuracy: 0.7187793427230047 . Test Accuracy: 0.681203007518797 .
# Epoch: 196, Loss: 1.3262
# Train Accuracy: 0.7281690140845071 . Test Accuracy: 0.6917293233082706 .
# Epoch: 198, Loss: 1.2942
# Train Accuracy: 0.7169014084507043 . Test Accuracy: 0.6842105263157895 . 
# Epoch: 200, Loss: 1.3041
# Train Accuracy: 0.6892018779342723 . Test Accuracy: 0.693609022556391 .

# Best Case:
# Epoch: 147, Loss: 1.3581
# Train Accuracy: 0.7370892018779343 . Test Accuracy: 0.7157894736842105 .




