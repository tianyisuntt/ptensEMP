import torch
import ptens
from Cora_models import *
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'train_rest',
                                  num_val = 433,
                                  num_test = 540)
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
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

"""
Dataset: Cora
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
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

    
model = P1GCN0(hidden_channels = 32, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print(f'Train acc:{train_acc:.4f}, Test acc:{test_acc:.4f}')
#print("Epoch:", epoch, ", Loss:" loss, ", Train Accuracy:", train_acc, ", Test Accuracy:", test_acc, ".")
print("=================================================================")
#"""
# lr=0.001, weight_decay=8e-4
# train/valid/test = 64/16/20 
# Epoch: 200, Loss: 1.3336
# Train Accuracy: 0.7230046948356808 . Test Accuracy: 0.6481203007518797 .


model = P1GCN(hidden_channels = 32, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print(f'Train acc:{train_acc:.4f}, Test acc:{test_acc:.4f}')
print("=================================================================")
# Epoch: 192, Loss: 1.9435
# Train acc:0.5167, Test acc: 0.4991

model = P1GCN1(hidden_channels = 32, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print(f'Train acc:{train_acc:.4f}, Test acc:{test_acc:.4f}')
#print("Epoch:", epoch, ", Loss:" loss, ", Train Accuracy:", train_acc, ", Test Accuracy:", test_acc, ".")
print("=================================================================")
# Epoch: 126, Loss: 1.9113
# Train acc:0.7889, Test acc:0.7839
# Epoch: 142, Loss: 2.5638
# Train acc:0.7990, Test acc:0.7944
# Epoch: 143, Loss: 1.9023
# Train acc:0.7852, Test acc:0.7816
# Epoch: 145, Loss: 1.8775
# Train acc:0.7815, Test acc:0.7666
# Epoch: 146, Loss: 1.7851
# Train acc:0.7981, Test acc:0.7850
# Epoch: 148, Loss: 1.9064
# Train acc:0.7977, Test acc:0.7852
# Epoch: 149, Loss: 1.7687
# Train acc:0.8063, Test acc:0.7926
# Epoch: 159, Loss: 1.9763
# Train acc:0.7963, Test acc:0.7856
# Epoch: 160, Loss: 2.6446
# Train acc:0.7885, Test acc:0.7870
# Epoch: 174, Loss: 3.1348
# Train acc:0.8000, Test acc:0.7963
# Epoch: 180, Loss: 3.6925
# Train acc:0.7926, Test acc:0.7833
# Epoch: 190, Loss: 2.3124
# Train acc:0.7815, Test acc:0.7752
# Epoch: 200, Loss: 6.3709
# Train acc:0.7850, Test acc:0.7870

# Best Case:
# Epoch: 149, Loss: 1.7687
# Train acc:0.8063, Test acc:0.7926

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x,data.G, None)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step() 
      return loss
def test():
      model.eval()
      out = model(data.x,data.G, None)
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
      val_correct = pred[data.val_mask] == data.y[data.val_mask]  
      val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, val_acc, test_acc
    
channels_in = dataset.num_features
out_channels = dataset.num_classes
convolution_dim = 64
dense_dim = 128
reduction_type = 'mean'
model = PCONV(channels_in, convolution_dim,dense_dim,out_channels)
lr = 0.01
wd = 8e-1
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}', ", Train Accuracy:", train_acc, ", Validation Accuracy", val_acc, ", Test Accuracy:", test_acc, ".")
print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Accuracy:', train_acc,", Validation Accuracy:", val_acc, ", Test Accuracy:", test_acc, ".")
print('=================================================================')




