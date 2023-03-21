import torch
import ptens
from Wisconsin_models import *
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
from torch_geometric.datasets import WebKB
dataset = WebKB(root='data/WebKB', name='Wisconsin', transform=NormalizeFeatures())
data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 40,
                                  num_test = 50)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)


def train():
      model.train()
      optimizer.zero_grad()
      out = model(ptens.ptensors0.from_matrix(data.x),data.G).torch()
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


model = PMLP(hidden_channels = 32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

   
model = P0GCN(hidden_channels = 64) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
ls = []
tr_ac = []
te_ac = []
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    ls.append(loss)
    tr_ac.append(train_acc)
    te_ac.append(test_acc)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print("Loss:", ls)
print("Train Accuracys:", tr_ac)
print("Test Accuracys:", te_ac)
print('=================================================================')


model = P1GCN(hidden_channels = 128, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
# hidden_channels = 64, reduction_type = "mean"
# lr=0.01, weight_decay=5e-4
# epoches = 200
# Epoch: 200, Loss: 1.3073
# Train Accuracy: 0.4658385093167702 . Test Accuracy: 0.42 .

# hidden_channels = 128, reduction_type = "mean"
# lr=0.001, weight_decay=8e-4
# Epoch: 200, Loss: 1.0864
# Train Accuracy: 0.515527950310559 . Test Accuracy: 0.42 .


model = P1GCN0(embedding_dim = 300, hidden_channels = 64, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')


model = P1GCN2(hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print(f'Train acc:{train_acc:.4f}, Test acc:{test_acc:.4f}')
print('=================================================================')
'''
Initializing ptens without GPU support.
Epoch: 001, Loss: 1.6046
Train acc:0.2422, Test acc:0.2200
Epoch: 002, Loss: 1.5845
Train acc:0.2871, Test acc:0.2800
Epoch: 003, Loss: 1.5787
Train acc:0.2671, Test acc:0.2000
Epoch: 004, Loss: 1.5630
Train acc:0.3857, Test acc:0.3000
Epoch: 005, Loss: 1.5554
Train acc:0.3416, Test acc:0.2800
Epoch: 006, Loss: 1.5542
Train acc:0.3168, Test acc:0.3000
Epoch: 007, Loss: 1.5236
Train acc:0.3540, Test acc:0.2600
Epoch: 008, Loss: 1.5425
Train acc:0.2857, Test acc:0.2800
Epoch: 009, Loss: 1.5328
Train acc:0.3851, Test acc:0.3200
Epoch: 010, Loss: 1.5042
Train acc:0.3478, Test acc:0.2800
Epoch: 011, Loss: 1.5305
Train acc:0.3540, Test acc:0.3200
Epoch: 012, Loss: 1.5204
Train acc:0.3540, Test acc:0.3200
Epoch: 013, Loss: 1.5268
Train acc:0.3430, Test acc:0.3400
Epoch: 014, Loss: 1.4883
Train acc:0.3478, Test acc:0.3000
Epoch: 015, Loss: 1.4871
Train acc:0.3478, Test acc:0.3000
Epoch: 016, Loss: 1.4575
Train acc:0.3727, Test acc:0.3400
Epoch: 017, Loss: 1.4554
Train acc:0.3727, Test acc:0.3600
Epoch: 018, Loss: 1.4688
Train acc:0.3230, Test acc:0.3400
Epoch: 019, Loss: 1.4642
Train acc:0.3292, Test acc:0.2800
Epoch: 020, Loss: 1.4724
Train acc:0.3665, Test acc:0.3600
Epoch: 021, Loss: 1.4610
Train acc:0.3230, Test acc:0.2800
Epoch: 022, Loss: 1.4598
Train acc:0.3168, Test acc:0.3000
Epoch: 023, Loss: 1.4747
Train acc:0.3230, Test acc:0.3200
Epoch: 024, Loss: 1.4398
Train acc:0.2422, Test acc:0.2200
Epoch: 025, Loss: 1.4244
Train acc:0.3665, Test acc:0.3600
Epoch: 026, Loss: 1.4543
Train acc:0.3727, Test acc:0.3400
Epoch: 027, Loss: 1.4193
Train acc:0.3478, Test acc:0.3600
Epoch: 028, Loss: 1.4291
Train acc:0.3416, Test acc:0.3200
Epoch: 029, Loss: 1.4102
Train acc:0.3851, Test acc:0.3400
Epoch: 030, Loss: 1.4030
Train acc:0.3416, Test acc:0.3600
Epoch: 031, Loss: 1.4300
Train acc:0.3789, Test acc:0.3600
Epoch: 032, Loss: 1.4548
Train acc:0.3851, Test acc:0.3800
Epoch: 033, Loss: 1.4572
Train acc:0.3665, Test acc:0.3400
Epoch: 034, Loss: 1.4131
Train acc:0.3416, Test acc:0.2800
Epoch: 035, Loss: 1.4061
Train acc:0.3975, Test acc:0.3800
Epoch: 036, Loss: 1.4003
Train acc:0.3789, Test acc:0.4000
Epoch: 037, Loss: 1.4000
Train acc:0.3665, Test acc:0.4000
Epoch: 038, Loss: 1.3808
Train acc:0.3913, Test acc:0.3800
Epoch: 039, Loss: 1.4133
Train acc:0.3478, Test acc:0.4000
Epoch: 040, Loss: 1.3698
Train acc:0.3665, Test acc:0.3400
Epoch: 041, Loss: 1.3933
Train acc:0.3851, Test acc:0.3200
Epoch: 042, Loss: 1.4071
Train acc:0.3540, Test acc:0.3600
Epoch: 043, Loss: 1.3673
Train acc:0.4286, Test acc:0.3400
Epoch: 044, Loss: 1.3824
Train acc:0.3975, Test acc:0.3600
Epoch: 045, Loss: 1.3876
Train acc:0.4348, Test acc:0.3800
Epoch: 046, Loss: 1.4214
Train acc:0.3851, Test acc:0.3800
Epoch: 047, Loss: 1.3545
Train acc:0.3789, Test acc:0.4000
Epoch: 048, Loss: 1.3695
Train acc:0.4224, Test acc:0.3800
Epoch: 049, Loss: 1.3708
Train acc:0.4037, Test acc:0.4600
Epoch: 050, Loss: 1.3559
Train acc:0.3602, Test acc:0.3600
Epoch: 051, Loss: 1.3358
Train acc:0.3789, Test acc:0.3800
Epoch: 052, Loss: 1.3599
Train acc:0.3913, Test acc:0.3800
Epoch: 053, Loss: 1.3509
Train acc:0.3789, Test acc:0.4000
Epoch: 054, Loss: 1.4234
Train acc:0.4286, Test acc:0.3400
Epoch: 055, Loss: 1.3770
Train acc:0.3789, Test acc:0.4400
Epoch: 056, Loss: 1.3437
Train acc:0.4099, Test acc:0.4200
Epoch: 057, Loss: 1.3460
Train acc:0.3789, Test acc:0.4000
Epoch: 058, Loss: 1.3292
Train acc:0.3602, Test acc:0.3200
Epoch: 059, Loss: 1.3000
Train acc:0.3416, Test acc:0.3200
Epoch: 060, Loss: 1.4204
Train acc:0.3665, Test acc:0.3600
Epoch: 061, Loss: 1.3456
Train acc:0.4037, Test acc:0.4400
Epoch: 062, Loss: 1.3350
Train acc:0.4099, Test acc:0.4600
Epoch: 063, Loss: 1.3470
Train acc:0.3727, Test acc:0.3800
Epoch: 064, Loss: 1.3217
Train acc:0.3665, Test acc:0.3800
Epoch: 065, Loss: 1.3515
Train acc:0.4224, Test acc:0.3600
Epoch: 066, Loss: 1.3630
Train acc:0.3727, Test acc:0.3800
Epoch: 067, Loss: 1.4020
Train acc:0.3975, Test acc:0.4000
Epoch: 068, Loss: 1.3103
Train acc:0.4410, Test acc:0.4600
Epoch: 069, Loss: 1.3783
Train acc:0.3789, Test acc:0.3800
Epoch: 070, Loss: 1.2783
Train acc:0.3789, Test acc:0.3400
Epoch: 071, Loss: 1.3268
Train acc:0.4410, Test acc:0.4000
Epoch: 072, Loss: 1.2937
Train acc:0.4037, Test acc:0.4000
Epoch: 073, Loss: 1.3234
Train acc:0.3789, Test acc:0.4000
Epoch: 074, Loss: 1.3023
Train acc:0.3913, Test acc:0.4600
Epoch: 075, Loss: 1.3251
Train acc:0.3851, Test acc:0.4000
Epoch: 076, Loss: 1.3671
Train acc:0.3975, Test acc:0.4600
Epoch: 077, Loss: 1.3277
Train acc:0.3727, Test acc:0.3600
Epoch: 078, Loss: 1.2815
Train acc:0.4286, Test acc:0.4200
Epoch: 079, Loss: 1.2566
Train acc:0.4099, Test acc:0.3800
Epoch: 080, Loss: 1.3039
Train acc:0.4224, Test acc:0.4000
Epoch: 081, Loss: 1.3008
Train acc:0.4410, Test acc:0.4800
Epoch: 082, Loss: 1.2865
Train acc:0.3727, Test acc:0.3400
Epoch: 083, Loss: 1.2938
Train acc:0.3975, Test acc:0.3800
Epoch: 084, Loss: 1.2763
Train acc:0.3851, Test acc:0.3800
Epoch: 085, Loss: 1.3481
Train acc:0.4348, Test acc:0.3800
Epoch: 086, Loss: 1.3340
Train acc:0.4348, Test acc:0.4000
Epoch: 087, Loss: 1.2956
Train acc:0.4789, Test acc:0.4000
Epoch: 088, Loss: 1.3045
Train acc:0.4472, Test acc:0.4600
Epoch: 089, Loss: 1.2809
Train acc:0.4461, Test acc:0.4400
Epoch: 090, Loss: 1.3599
Train acc:0.4596, Test acc:0.4200
Epoch: 091, Loss: 1.3153
Train acc:0.4224, Test acc:0.4600
Epoch: 092, Loss: 1.3329
Train acc:0.4472, Test acc:0.3800
Epoch: 093, Loss: 1.3020
Train acc:0.4534, Test acc:0.4600
Epoch: 094, Loss: 1.3115
Train acc:0.4658, Test acc:0.4400
Epoch: 095, Loss: 1.2958
Train acc:0.4534, Test acc:0.4600
Epoch: 096, Loss: 1.3293
Train acc:0.3106, Test acc:0.3400
Epoch: 097, Loss: 1.2826
Train acc:0.5037, Test acc:0.5000
Epoch: 098, Loss: 1.2408
Train acc:0.4472, Test acc:0.4400
Epoch: 099, Loss: 1.2847
Train acc:0.4161, Test acc:0.4200
Epoch: 100, Loss: 1.2987
Train acc:0.4161, Test acc:0.4000
Epoch: 101, Loss: 1.3302
Train acc:0.4410, Test acc:0.4800
Epoch: 102, Loss: 1.2729
Train acc:0.3913, Test acc:0.4600
Epoch: 103, Loss: 1.2548
Train acc:0.4596, Test acc:0.4400
Epoch: 104, Loss: 1.2453
Train acc:0.4720, Test acc:0.4400
Epoch: 105, Loss: 1.3362
Train acc:0.4410, Test acc:0.4000
Epoch: 106, Loss: 1.2704
Train acc:0.4099, Test acc:0.4200
Epoch: 107, Loss: 1.2669
Train acc:0.3727, Test acc:0.4200
Epoch: 108, Loss: 1.3073
Train acc:0.4783, Test acc:0.4800
Epoch: 109, Loss: 1.3014
Train acc:0.4783, Test acc:0.4600
Epoch: 110, Loss: 1.2480
Train acc:0.4658, Test acc:0.4400
Epoch: 111, Loss: 1.2364
Train acc:0.4286, Test acc:0.4600
Epoch: 112, Loss: 1.3148
Train acc:0.4410, Test acc:0.4800
Epoch: 113, Loss: 1.3599
Train acc:0.4845, Test acc:0.4800
Epoch: 114, Loss: 1.2255
Train acc:0.4348, Test acc:0.4400
Epoch: 115, Loss: 1.2567
Train acc:0.4783, Test acc:0.4200
Epoch: 116, Loss: 1.2885
Train acc:0.4969, Test acc:0.4600
Epoch: 117, Loss: 1.2655
Train acc:0.4472, Test acc:0.4200
Epoch: 118, Loss: 1.2630
Train acc:0.4224, Test acc:0.3600
Epoch: 119, Loss: 1.2865
Train acc:0.4224, Test acc:0.4400
Epoch: 120, Loss: 1.2049
Train acc:0.4720, Test acc:0.4200
Epoch: 121, Loss: 1.1900
Train acc:0.4161, Test acc:0.4400
Epoch: 122, Loss: 1.2785
Train acc:0.4037, Test acc:0.4400
Epoch: 123, Loss: 1.2700
Train acc:0.4348, Test acc:0.4600
Epoch: 124, Loss: 1.2278
Train acc:0.4720, Test acc:0.4800
Epoch: 125, Loss: 1.2719
Train acc:0.4286, Test acc:0.4600
Epoch: 126, Loss: 1.2514
Train acc:0.3913, Test acc:0.4000
Epoch: 127, Loss: 1.2556
Train acc:0.4224, Test acc:0.4400
Epoch: 128, Loss: 1.2282
Train acc:0.4472, Test acc:0.4600
Epoch: 129, Loss: 1.2817
Train acc:0.4037, Test acc:0.4000
Epoch: 130, Loss: 1.2034
Train acc:0.4720, Test acc:0.4600
Epoch: 131, Loss: 1.2234
Train acc:0.4596, Test acc:0.4400
Epoch: 132, Loss: 1.2202
Train acc:0.4720, Test acc:0.5000
Epoch: 133, Loss: 1.2548
Train acc:0.4099, Test acc:0.4000
Epoch: 134, Loss: 1.1881
Train acc:0.4534, Test acc:0.4200
Epoch: 135, Loss: 1.2595
Train acc:0.4099, Test acc:0.4400
Epoch: 136, Loss: 1.2011
Train acc:0.4410, Test acc:0.4800
Epoch: 137, Loss: 1.1732
Train acc:0.4969, Test acc:0.4600
Epoch: 138, Loss: 1.2126
Train acc:0.4720, Test acc:0.4200
Epoch: 139, Loss: 1.1934
Train acc:0.4534, Test acc:0.4800
Epoch: 140, Loss: 1.2339
Train acc:0.4907, Test acc:0.4800
Epoch: 141, Loss: 1.1924
Train acc:0.4658, Test acc:0.4600
Epoch: 142, Loss: 1.2739
Train acc:0.4348, Test acc:0.4400
Epoch: 143, Loss: 1.2020
Train acc:0.4224, Test acc:0.4400
Epoch: 144, Loss: 1.1672
Train acc:0.4969, Test acc:0.4600
Epoch: 145, Loss: 1.1885
Train acc:0.4658, Test acc:0.4600
Epoch: 146, Loss: 1.2266
Train acc:0.4596, Test acc:0.5000
Epoch: 147, Loss: 1.2019
Train acc:0.4845, Test acc:0.4800
Epoch: 148, Loss: 1.2900
Train acc:0.4783, Test acc:0.4600
Epoch: 149, Loss: 1.2142
Train acc:0.4596, Test acc:0.5000
Epoch: 150, Loss: 1.1797
Train acc:0.4534, Test acc:0.4200
Epoch: 151, Loss: 1.2115
Train acc:0.4224, Test acc:0.4600
Epoch: 152, Loss: 1.1937
Train acc:0.4845, Test acc:0.5000
Epoch: 153, Loss: 1.2717
Train acc:0.4907, Test acc:0.4800
Epoch: 154, Loss: 1.3011
Train acc:0.4658, Test acc:0.4400
Epoch: 155, Loss: 1.1826
Train acc:0.4658, Test acc:0.4600
Epoch: 156, Loss: 1.1694
Train acc:0.4969, Test acc:0.4800
Epoch: 157, Loss: 1.2591
Train acc:0.4783, Test acc:0.4600
Epoch: 158, Loss: 1.2039
Train acc:0.4783, Test acc:0.4200
Epoch: 159, Loss: 1.2310
Train acc:0.4410, Test acc:0.4200
Epoch: 160, Loss: 1.1567
Train acc:0.4099, Test acc:0.4400
Epoch: 161, Loss: 1.2010
Train acc:0.4596, Test acc:0.4800
Epoch: 162, Loss: 1.1629
Train acc:0.4845, Test acc:0.4400
Epoch: 163, Loss: 1.3103
Train acc:0.5093, Test acc:0.4400
Epoch: 164, Loss: 1.2268
Train acc:0.4845, Test acc:0.4800
Epoch: 165, Loss: 1.2992
Train acc:0.4845, Test acc:0.4400
Epoch: 166, Loss: 1.2146
Train acc:0.4037, Test acc:0.4000
Epoch: 167, Loss: 1.1573
Train acc:0.4845, Test acc:0.4800
Epoch: 168, Loss: 1.2069
Train acc:0.4783, Test acc:0.5000
Epoch: 169, Loss: 1.1699
Train acc:0.4410, Test acc:0.4400
Epoch: 170, Loss: 1.1598
Train acc:0.5031, Test acc:0.4400
Epoch: 171, Loss: 1.2239
Train acc:0.4658, Test acc:0.5000
Epoch: 172, Loss: 1.2280
Train acc:0.4410, Test acc:0.4200
Epoch: 173, Loss: 1.1947
Train acc:0.4845, Test acc:0.4800
Epoch: 174, Loss: 1.2117
Train acc:0.5031, Test acc:0.4600
Epoch: 175, Loss: 1.2049
Train acc:0.4720, Test acc:0.4400
Epoch: 176, Loss: 1.1898
Train acc:0.4658, Test acc:0.4400
Epoch: 177, Loss: 1.1400
Train acc:0.4410, Test acc:0.5000
Epoch: 178, Loss: 1.2894
Train acc:0.3354, Test acc:0.3400
Epoch: 179, Loss: 1.1732
Train acc:0.4720, Test acc:0.4800
Epoch: 180, Loss: 1.2044
Train acc:0.4720, Test acc:0.4400
Epoch: 181, Loss: 1.2187
Train acc:0.4983, Test acc:0.5000
Epoch: 182, Loss: 1.2285
Train acc:0.4348, Test acc:0.4000
Epoch: 183, Loss: 1.1853
Train acc:0.5093, Test acc:0.4800
Epoch: 184, Loss: 1.1840
Train acc:0.5093, Test acc:0.4600
Epoch: 185, Loss: 1.1695
Train acc:0.4534, Test acc:0.4800
Epoch: 186, Loss: 1.1801
Train acc:0.4783, Test acc:0.4800
Epoch: 187, Loss: 1.1368
Train acc:0.4658, Test acc:0.4600
Epoch: 188, Loss: 1.2028
Train acc:0.4907, Test acc:0.4400
Epoch: 189, Loss: 1.1894
Train acc:0.4658, Test acc:0.4800
Epoch: 190, Loss: 1.2336
Train acc:0.4783, Test acc:0.4600
Epoch: 191, Loss: 1.1959
Train acc:0.4907, Test acc:0.4800
Epoch: 192, Loss: 1.1347
Train acc:0.4037, Test acc:0.4000
Epoch: 193, Loss: 1.1560
Train acc:0.4969, Test acc:0.4800
Epoch: 194, Loss: 1.1362
Train acc:0.5031, Test acc:0.4600
Epoch: 195, Loss: 1.2061
Train acc:0.4658, Test acc:0.4600
Epoch: 196, Loss: 1.1501
Train acc:0.4907, Test acc:0.4600
Epoch: 197, Loss: 1.1331
Train acc:0.4969, Test acc:0.4600
Epoch: 198, Loss: 1.1986
Train acc:0.4845, Test acc:0.4600
Epoch: 199, Loss: 1.0900
Train acc:0.5093, Test acc:0.5000
Epoch: 200, Loss: 1.1278
Train acc:0.4783, Test acc:0.4600

Best Case:
Epoch: 199, Loss: 1.0900
Train acc:0.5093, Test acc:0.5000
'''


