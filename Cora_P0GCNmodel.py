import torch
import ptens
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 500,
                                  num_val = 300)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)


class P0GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_0P(dataset.num_features, hidden_channels)
        self.conv2 = ptens.modules.ConvolutionalLayer_0P(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5)

    def forward(self, x, edge_index):
      #  x = ptens.ptensors0.from_matrix(x)
       # edge_index = edge_index.type(torch.FloatTensor)
       # edge_index = ptens.graph.from_matrix(edge_index)
        
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
      #  x = x.torch()
        return x

def train():
      model.train()
      optimizer.zero_grad()
     # print(data.x)
    #  print(data.G)
      out = model(ptens.ptensors0.from_matrix(data.x),data.G).torch()
    #  out = out.torch()
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

    
model = P0GCN(hidden_channels = 32) # subject to change
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
hidden_channels = 32
epoches = 200
Train Accuracy: 0.8652719665271966 . Test Accuracy: 0.9444444444444444 .
"""