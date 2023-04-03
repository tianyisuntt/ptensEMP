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
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 532,
                                  num_test = 665)
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

