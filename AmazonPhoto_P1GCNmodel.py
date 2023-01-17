import torch
import ptens
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Amazon(root='data/Amazon', name='Photo', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 1250, 
                                  num_val = 500)  
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)

class P1GCN(torch.nn.Module):
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
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

    
model = P1GCN(hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
"""
AmazonPhoto:
num_train_per_class = 1250, num_val = 500
hidden_channels = 32, reduction_type = "mean"
epoches = 200
Epoch: 001, Loss: 2476.0469
Epoch: 002, Loss: 22032.2812
Epoch: 003, Loss: 6701.9839
Epoch: 004, Loss: 15550.0723
Epoch: 005, Loss: 14954.2910
Epoch: 006, Loss: 17261.1895
Epoch: 007, Loss: 68228.0391
Epoch: 008, Loss: 48328.3359
Epoch: 009, Loss: 36422.9727
Epoch: 010, Loss: 18651.6777
Epoch: 011, Loss: 22343.1816
Epoch: 012, Loss: 29120.5625
Epoch: 013, Loss: 7911.2900
Epoch: 014, Loss: 24134.6660
Epoch: 015, Loss: 20640.5293
Epoch: 016, Loss: 19298.7832
Epoch: 017, Loss: 20599.6992
Epoch: 018, Loss: 36007.5234
Epoch: 019, Loss: 26406.3770
Epoch: 020, Loss: 39741.9219
Epoch: 021, Loss: 36061.4375
Epoch: 022, Loss: 28949.1777
Epoch: 023, Loss: 50308.0273
Epoch: 024, Loss: 93904.5000
Epoch: 025, Loss: 58642.6914
Epoch: 026, Loss: 47052.0000
Epoch: 027, Loss: 70689.3125
Epoch: 028, Loss: 29113.6074
Epoch: 029, Loss: 38574.0742
Epoch: 030, Loss: 12687.5342
Epoch: 031, Loss: 23917.0391
Epoch: 032, Loss: 22722.3379
Epoch: 033, Loss: 30001.3789
Epoch: 034, Loss: 9143.5342
Epoch: 035, Loss: 17530.5801
Epoch: 036, Loss: 47337.2461
Epoch: 037, Loss: 28136.9238
Epoch: 038, Loss: 82084.2344
Epoch: 039, Loss: 60338.9102
Epoch: 040, Loss: 52575.8711
Epoch: 041, Loss: 53892.5000
Epoch: 042, Loss: 29572.5547
Epoch: 043, Loss: 35241.6211
Epoch: 044, Loss: 20382.3086
Epoch: 045, Loss: 22127.1758
Epoch: 046, Loss: 36473.8672
Epoch: 047, Loss: 32047.8691
Epoch: 048, Loss: 10720.2295
Epoch: 049, Loss: 34601.4336
Epoch: 050, Loss: 27219.7129
Epoch: 051, Loss: 11621.7393
Epoch: 052, Loss: 59088.4883
Epoch: 053, Loss: 37054.9883
Epoch: 054, Loss: 29076.8906
Epoch: 055, Loss: 14516.3857
Epoch: 056, Loss: 21211.9531
Epoch: 057, Loss: 11086.6777
Epoch: 058, Loss: 28503.8008
Epoch: 059, Loss: 18851.5117
Epoch: 060, Loss: 27427.6250
Epoch: 061, Loss: 18856.9590
Epoch: 062, Loss: 40541.1367
Epoch: 063, Loss: 32326.3203
Epoch: 064, Loss: 15082.2227
Epoch: 065, Loss: 19369.9023
Epoch: 066, Loss: 21169.2402
Epoch: 067, Loss: 18482.6055
Epoch: 068, Loss: 11986.3008
Epoch: 069, Loss: 15649.6768
Epoch: 070, Loss: 20809.5508
Epoch: 071, Loss: 18442.1074
Epoch: 072, Loss: 13861.1035
Epoch: 073, Loss: 20370.2402
Epoch: 074, Loss: 43719.9922
Epoch: 075, Loss: 25203.8477
Epoch: 076, Loss: 40603.3047
Epoch: 077, Loss: 25901.1543
Epoch: 078, Loss: 26658.9746
Epoch: 079, Loss: 22463.1074
Epoch: 080, Loss: 7812.1255
Epoch: 081, Loss: 29199.5859
Epoch: 082, Loss: 12262.0781
Epoch: 083, Loss: 4213.2339
Epoch: 084, Loss: 23253.9434
Epoch: 085, Loss: 25155.5840
Epoch: 086, Loss: 20027.3887
Epoch: 087, Loss: 7923.7314
Epoch: 088, Loss: 28593.8867
Epoch: 089, Loss: 18944.5977
Killed: 9
"""
