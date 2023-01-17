import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
dataset = WebKB(root='data/WebKB', name='Wisconsin', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.
transform_nodes = RandomNodeSplit(split = 'random', 
                                 num_train_per_class = 100,
                                 num_val = 1, 
                                 num_test = 150)
data = transform_nodes(data)

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels1):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels1)
        self.lin3 = Linear(hidden_channels1, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels1):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels1)
        self.conv3 = GCNConv(hidden_channels1, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GAT(torch.nn.Module): 
    def __init__(self, hidden_channels1):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels1)
        self.conv4 = GATConv(hidden_channels1, dataset.num_classes)
        
    def forward(self, x, edge_index, return_attention_weights = True):
        x, w1 = self.conv1(x, edge_index, return_attention_weights = True)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x, w2 = self.conv4(x, edge_index, return_attention_weights = True)
        return x


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
    

print(f'Dataset: {dataset}:')
print(data)
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print('=================================================================')

model = GAT(hidden_channels1 = 16)
# 256: Train Accuracy: 0.8412017167381974 . Test Accuracy: 0.6470588235294118 .
# 64: Train Accuracy: 0.7339055793991416 . Test Accuracy: 0.7647058823529411 .
# 16: Train Accuracy: 0.5793991416309013 . Test Accuracy: 0.8235294117647058 .
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

model = GCN(hidden_channels1 = 64)
# 16: Train Accuracy: 0.5536480686695279 . Test Accuracy: 0.7058823529411765 .
# 64: Train Accuracy: 0.6137339055793991 . Test Accuracy: 0.9411764705882353 .
# 64: Train Accuracy: 0.6008583690987125 . Test Accuracy: 0.8235294117647058 .
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

model = MLP(hidden_channels1 = 32)
# 16: Train Accuracy: 0.8927038626609443 . Test Accuracy: 1.0 .
# 32: Train Accuracy: 0.9484978540772532 . Test Accuracy: 0.9411764705882353 .
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

"""
Dataset: wisconsin():
Data(x=[251, 1703], edge_index=[2, 515], y=[251], train_mask=[251], val_mask=[251], test_mask=[251])
Number of graphs: 1
Number of features: 1703
Number of classes: 5
Number of nodes: 251
Number of edges: 515
Average node degree: 2.05
Number of training nodes: 233
Training node label rate: 0.93
Has isolated nodes: False
Has self-loops: True
Is undirected: False
=================================================================
GAT(
  (conv1): GATConv(1703, 16, heads=1)
  (conv4): GATConv(16, 5, heads=1)
)
Train Accuracy: 0.5965665236051502 . Test Accuracy: 0.8235294117647058 .
=================================================================
GCN(
  (conv1): GCNConv(1703, 64)
  (conv3): GCNConv(64, 5)
)
Train Accuracy: 0.6137339055793991 . Test Accuracy: 0.8235294117647058 .
=================================================================
MLP(
  (lin1): Linear(in_features=1703, out_features=32, bias=True)
  (lin3): Linear(in_features=32, out_features=5, bias=True)
)
Train Accuracy: 0.9484978540772532 . Test Accuracy: 0.9411764705882353 .
=================================================================
"""
