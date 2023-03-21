import torch
import ptens
from Cornell_models import *
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from torch_geometric.datasets import WebKB
from Transforms import PreComputeNorm
dataset = WebKB(root='data/WebKB', name='Cornell', transform=NormalizeFeatures())
data = dataset[0]
Normalization = PreComputeNorm()
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 29,
                                  num_test = 36)
data = Normalization(data)
data = transform_nodes(data)

def train():
      model.train()
      optimizer.zero_grad()  
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step() 
      return loss
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, test_acc

    
model = PMLP(hidden_channels1=168, hidden_channels2=168, hidden_channels3=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
"""
Cornell
plit = 'random', num_train_per_class = 70, num_val = 3, num_test = 110
dropout prob = 0.6
hidden_channels1=168, hidden_channels2=168, hidden_channels3=32
epoches = 200
Train Accuracy: 0.9941520467836257 . Test Accuracy: 0.8888888888888888 .
"""
    
model = P0GCN(hidden_channels = 256) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 601):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')


model = P1GCN(hidden_channels = 64, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
# Epoch: 173, Loss: 1.5670
# Train Accuracy: 0.6708333333333333 . Test Accuracy: 0.5888888888888889 .


model = P1GCN0(hidden_channels1 = 256, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

    
model = P1GCN2(embedding_dim = 256, hidden_channels = 128, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 601):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
# embedding_dim = 256, hidden_channels = 128, reduction_type = "mean"
# lr=0.001, weight_decay=8e-4
#Epoch: 249, Loss: 1.6202
#Train Accuracy: 0.2457627118644068 . Test Accuracy: 0.2222222222222222 .

