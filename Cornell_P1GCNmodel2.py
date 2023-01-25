import torch
import ptens
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch, PreComputeNorm
dataset = WebKB(root='data/WebKB', name='Cornell', transform=NormalizeFeatures())
data = dataset[0]

Normalization = PreComputeNorm()
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 29,
                                  num_test = 36)
on_learn_transform = ToPtens_Batch()

data = Normalization(data)
data = transform_nodes(data)
data = on_learn_transform(data)

class P1GCN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = ptens.modules.Linear(dataset.num_features, embedding_dim)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(embedding_dim, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, hidden_channels, reduction_type)
        self.conv3 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, hidden_channels, reduction_type)
        self.batchnorm = ptens.modules.BatchNorm(hidden_channels) 
        self.lin2 = ptens.modules.Linear(hidden_channels, dataset.num_classes)
        self.lin3 = ptens.modules.Linear(dataset.num_classes, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x, False)
        x = self.lin1(x).relu()
        x = self.dropout(x)
        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        x = self.conv3(x,edge_index)
        x = self.batchnorm(x)
        x = self.lin2(x).relu()
        x = self.dropout(x)
        x = self.lin3(x)
        x = ptens.linmaps0(x, False)
        return x

def train():
      model.train()
      optimizer.zero_grad()
      data_x = ptens.ptensors0.from_matrix(data.x)
      out = model(data_x,data.G).torch()
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward(retain_graph=True) 
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

    
model = P1GCN(embedding_dim = 256, hidden_channels = 128, reduction_type = "mean") 
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
