import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
dataset = WebKB(root='data/WebKB', name='Cornell', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'random', 
                                 num_train_per_class = 70,
                                 num_val = 3, 
                                 num_test = 110)
data = transform_nodes(data)

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels1, hidden_channels2, hidden_channels3):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels1)
        self.lin2 = Linear(hidden_channels1, hidden_channels2)
        self.lin3 = Linear(hidden_channels2, hidden_channels3)
        self.lin4 = Linear(hidden_channels3, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin2(x)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin3(x)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin4(x)
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

model = MLP(hidden_channels1=168, hidden_channels2=168, hidden_channels3=32)
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
Dataset: cornell():
Data(x=[183, 1703], edge_index=[2, 298], y=[183], train_mask=[183], val_mask=[183], test_mask=[183])
Number of graphs: 1
Number of features: 1703
Number of classes: 5
Number of nodes: 183
Number of edges: 298
Average node degree: 1.63
Number of training nodes: 171
Training node label rate: 0.93
Has isolated nodes: False
Has self-loops: True
Is undirected: False
=================================================================
MLP(
  (lin1): Linear(in_features=1703, out_features=168, bias=True)
  (lin2): Linear(in_features=168, out_features=168, bias=True)
  (lin3): Linear(in_features=168, out_features=32, bias=True)
  (lin4): Linear(in_features=32, out_features=5, bias=True)
)
dropout prob = 0.5
epoches = 200
Train Accuracy: 0.9005847953216374 . Test Accuracy: 0.8888888888888888 .
=================================================================
V
dropout prob = 0.6
epoches = 200
Train Accuracy: 0.9122807017543859 . Test Accuracy: 0.8888888888888888 .
"""
