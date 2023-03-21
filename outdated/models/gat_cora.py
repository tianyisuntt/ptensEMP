from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
from torch_geometric.transforms.random_node_split import RandomNodeSplit
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 500,
                                  num_val = 300)
data = transform_nodes(data)

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module): 
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index, reture_attention_weights = True):
        print('x', x.shape)
        x, w1 = self.conv1(x, edge_index, return_attention_weights = True)
        print('x.conv1', x.shape)
        x = x.relu()
        print('x.relu()', x.shape)
        #x = F.dropout(x, p=0.5, training=self.training)
        #print('x.dropout', x.shape)
        x, w2 = self.conv2(x, edge_index, return_attention_weights = True)
        print('x.conv2', x.shape)
        return x

def train():
    model.train()
    optimizer.zero_grad() # clear gradients
    out = model(data.x, data.edge_index) # perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # compute the loss solely based on the training nodes.
    loss.backward() # derive gradients
    optimizer.step() # update parameters based on gradients
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim = 1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return train_acc, test_acc


### Train the model
model = GAT(hidden_channels = 256)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(1, 2):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    #print("Epoch:",epoch, ". Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")

print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")



