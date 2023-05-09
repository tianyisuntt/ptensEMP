import torch
import ptens
from AmazonPhoto_models import *
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Amazon(root='data/Amazon', name='Photo', transform=NormalizeFeatures())
data = dataset[0] 
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 1224,
                                  num_test = 1530)  
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
AmazonPhoto:
num_train_per_class = 1250, num_val = 500
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

model = P1GCN(hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
"""
AmazonPhoto:
num_train_per_class = 1250, num_val = 500
hidden_channels = 32, reduction_type = "mean"
epoches = 200

Killed: 9
"""

model = P1GCN2(embedding_dim = 64, hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
# Epoch: 048, Loss: 16118.5820
# Train Accuracy: 0.24325980392156862 . Test Accuracy: 0.2797385620915033 .
# killed: 9

