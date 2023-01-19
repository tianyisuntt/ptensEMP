import torch
import ptens
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch, PreComputeNorm
dataset = WebKB(root='data/WebKB', name='Cornell', transform=NormalizeFeatures())
data = dataset[0]

Normalization = PreComputeNorm()
transform_nodes = RandomNodeSplit(split = 'random', 
                                  num_train_per_class = 70,
                                  num_val = 3,
                                  num_test = 110)
on_learn_transform = ToPtens_Batch()

data = Normalization(data)
data = transform_nodes(data)
data = on_learn_transform(data)

class P1GCN(torch.nn.Module):
    def __init__(self, hidden_channels1, hidden_channels2, hidden_channels3, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels1, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels1, hidden_channels2, reduction_type)
        self.conv3 = ptens.modules.ConvolutionalLayer_1P(hidden_channels2, hidden_channels3, reduction_type)
        self.conv4 = ptens.modules.ConvolutionalLayer_1P(hidden_channels3, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.6,device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = ptens.linmaps0(x)
        return x

def train():
      model.train()
      optimizer.zero_grad()
      data_x = ptens.ptensors0.from_matrix(data.x)
      out = model(data_x,data.G).torch()
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step() 
      return loss
def test():
      model.eval()
      data_x = ptens.ptensors0.from_matrix(data.x)
      out = model(data_x,data.G).torch()
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, test_acc

    
model = P1GCN(hidden_channels1 = 256, hidden_channels2 = 128, hidden_channels3 = 64, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 601):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
"""
Cornell
round1:
num_train_per_class = 70, num_val = 3, num_test = 110
dropout prob = 0.6
hidden_channels = 256, reduction_type = "mean"
epoches = 600
Train Accuracy: 0.38011695906432746 . Test Accuracy: 0.0 .
=================================================================
round2:
num_train_per_class = 70, num_val = 3, num_test = 110
dropout prob = 0.6
hidden_channels = 256, reduction_type = "sum"
epoches = 600
Train Accuracy: 0.3508771929824561 . Test Accuracy: 0.0 .
=================================================================
round3:
num_train_per_class = 70, num_val = 3, num_test = 110
dropout prob = 0.6
hidden_channels1 = 256, hidden_channels2 = 128, hidden_channels3 = 64, reduction_type = "mean"
epoches = 600
Train Accuracy: 0.26900584795321636 . Test Accuracy: 0.0 .
"""
