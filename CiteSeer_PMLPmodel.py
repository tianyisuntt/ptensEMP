import torch
import ptens
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit

dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 510,
                                  num_val = 500)
data = transform_nodes(data)


class PMLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.plin1 = ptens.modules.Linear(dataset.num_features, hidden_channels)
        self.plin2 = ptens.modules.Linear(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)

    def forward(self, x, edge_index):
        x = ptens.ptensors0.from_matrix(x)
        x = self.plin1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.plin2(x)
        x = x.torch()
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
Train Accuracy: 0.7339971550497866 . Test Accuracy: 0.8666666666666667 .
"""
