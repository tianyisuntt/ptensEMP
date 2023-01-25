import torch
import ptens
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0] 
transform_nodes = RandomNodeSplit(split = 'train_rest',
                                  num_val = 433,
                                  num_test = 540)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)

class P1GCN(torch.nn.Module):
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x, False)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = ptens.linmaps0(x, False).torch()
        x = F.log_softmax(x, dim=1)
        x = ptens.ptensors0.from_matrix(x)
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

    
model = P1GCN(hidden_channels = 32, reduction_type = "mean") 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print(f'Train acc:{train_acc:.4f}, Test acc:{test_acc:.4f}')
#print("Epoch:", epoch, ", Loss:" loss, ", Train Accuracy:", train_acc, ", Test Accuracy:", test_acc, ".")
print("=================================================================")
# Epoch: 126, Loss: 1.9113
# Train acc:0.7889, Test acc:0.7839
# Epoch: 142, Loss: 2.5638
# Train acc:0.7990, Test acc:0.7944
# Epoch: 143, Loss: 1.9023
# Train acc:0.7852, Test acc:0.7816
# Epoch: 145, Loss: 1.8775
# Train acc:0.7815, Test acc:0.7666
# Epoch: 146, Loss: 1.7851
# Train acc:0.7981, Test acc:0.7850
# Epoch: 148, Loss: 1.9064
# Train acc:0.7977, Test acc:0.7852
# Epoch: 149, Loss: 1.7687
# Train acc:0.8063, Test acc:0.7926
# Epoch: 159, Loss: 1.9763
# Train acc:0.7963, Test acc:0.7856
# Epoch: 160, Loss: 2.6446
# Train acc:0.7885, Test acc:0.7870
# Epoch: 174, Loss: 3.1348
# Train acc:0.8000, Test acc:0.7963
# Epoch: 180, Loss: 3.6925
# Train acc:0.7926, Test acc:0.7833
# Epoch: 190, Loss: 2.3124
# Train acc:0.7815, Test acc:0.7752
# Epoch: 200, Loss: 6.3709
# Train acc:0.7850, Test acc:0.7870

# Best Case:
# Epoch: 149, Loss: 1.7687
# Train acc:0.8063, Test acc:0.7926
