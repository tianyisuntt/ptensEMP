import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 510,
                                  num_val = 500)
data = transform_nodes(data)

class MLP(torch.nn.Module):
    #def __init__(self, hidden_channels1, hidden_channels2, hidden_channels3):
    def __init__(self, hidden_channels1):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels1)
      #  self.lin2 = Linear(hidden_channels1, hidden_channels2)
      #  self.lin3 = Linear(hidden_channels2, hidden_channels3)
      #  self.lin4 = Linear(hidden_channels3, dataset.num_classes)
        self.lin2 = Linear(hidden_channels1, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
       # x = x.relu()
       # x = F.dropout(x, p=0.5, training=self.training)
       # x = self.lin3(x)
       # x = x.relu()
       # x = F.dropout(x, p=0.5, training=self.training)
       # x = self.lin4(x)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module): 
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index, return_attention_weights = True):
        x, w1 = self.conv1(x, edge_index, return_attention_weights = True)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x, w2 = self.conv2(x, edge_index, return_attention_weights = True)
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

model = GAT(hidden_channels = 256)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

model = GCN(hidden_channels=16)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

#model = MLP(hidden_channels1=256, hidden_channels2=168, hidden_channels3=32)
model = MLP(hidden_channels1=256)
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
Dataset: CiteSeer():
Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])
Number of graphs: 1
Number of features: 3703
Number of classes: 6
Number of nodes: 3327
Number of edges: 9104
Average node degree: 2.74
Number of training nodes: 2812
Training node label rate: 0.85
Has isolated nodes: True
Has self-loops: False
Is undirected: True
=================================================================
GAT(
  (conv1): GATConv(3703, 256, heads=1)
  (conv2): GATConv(256, 6, heads=1)
)
Train Accuracy: 0.8559743954480796 . Test Accuracy: 0.8666666666666667 .
=================================================================
GCN(
  (conv1): GCNConv(3703, 16)
  (conv2): GCNConv(16, 6)
)
Train Accuracy: 0.786628733997155 . Test Accuracy: 0.8 .
=================================================================
round1:
MLP(
  (lin1): Linear(in_features=3703, out_features=256, bias=True)
  (lin2): Linear(in_features=256, out_features=168, bias=True)
  (lin3): Linear(in_features=168, out_features=32, bias=True)
  (lin4): Linear(in_features=32, out_features=6, bias=True)
)
Train Accuracy: 0.9964438122332859 . Test Accuracy: 0.7333333333333333 .
=================================================================
round2:
MLP(
  (lin1): Linear(in_features=3703, out_features=256, bias=True)
  (lin2): Linear(in_features=256, out_features=6, bias=True)
)
Train Accuracy: 0.9406116642958748 . Test Accuracy: 0.8 .
=================================================================
"""
