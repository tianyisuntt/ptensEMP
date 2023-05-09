import torch
import ptens
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
data = dataset[0]
transform_nodes = RandomNodeSplit(split = 'train_rest', 
                                  num_val = 532,
                                  num_test = 665)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)


class ConvolutionalLayer1(torch.nn.Module):
  def __init__(self, channels_in: int, channels_out: int, hop_count: int) -> None:
    super().__init__()
    #assert channels_out % 2 == 0
    self.lin1 = torch.nn.Linear(2*channels_in,channels_in*4)
    self.norm1 = torch.nn.BatchNorm1d(channels_in*4)
    self.lin2 = torch.nn.Linear(channels_in*4,channels_out)
    self.norm2 = torch.nn.BatchNorm1d(channels_out)
    #self.drop = torch.nn.Dropout(dropout)
    self.nhoods = hop_count
  def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
    # we perform first order convolution
    x = x.transfer1(G.nhoods(self.nhoods),G,True)
    atoms = x.get_atoms()
    x = x.torch()
    x = torch.nn.functional.relu(self.norm1(self.lin1(x)),True)
    x = torch.nn.functional.relu(self.norm2(self.lin2(x)),True)
    #x = self.drop(x)
    x = ptens.ptensors1.from_matrix(x,atoms)
    # we perform zeroth order convolution
    # we update 1st order representation
    # we return updated representations
    return x
class ConvolutionalLayer0(torch.nn.Module):
  def __init__(self, channels_in: int, channels_out: int, hop_count: int) -> None:
    super().__init__()
    #assert channels_out % 2 == 0
    self.lin1_1 = torch.nn.Linear(channels_in,channels_in*2,bias=False)
    self.lin1_2 = torch.nn.Linear(channels_in,channels_in*2,bias=False)
    self.lin1_3 = torch.nn.Linear(channels_in,channels_in*2)
    self.norm1 = torch.nn.BatchNorm1d(channels_in*2)
    self.lin2 = torch.nn.Linear(channels_in*2,channels_out,bias=False)
    self.lin3 = ptens.modules.Linear(channels_out*2,channels_out)
    self.norm2 = torch.nn.BatchNorm1d(channels_out)
    #self.drop = torch.nn.Dropout(dropout)
    self.nhoods = hop_count
  def forward(self, x0: ptens.ptensors0, x1: ptens.ptensors1, G: ptens.graph) -> Tuple[ptens.ptensors0,ptens.ptensors1]:
    piece_1 = x0.torch()
    piece_2 = x0.gather(G,True).torch()
    piece_3 = x1.linmaps0(True).torch()
    atoms = x1.get_atoms()
    #
    result = self.lin1_1(piece_1) + self.lin1_2(piece_2) + self.lin1_3(piece_3)
    result = torch.nn.functional.relu(self.norm1(result),True)
    result = torch.nn.functional.relu(self.norm2(self.lin2(result)),True)
    #result = self.drop(result)
    #
    x0 = ptens.ptensors0.from_matrix(result)
    #
    adding = ptens.relu(self.lin3(x0.linmaps1(True).transfer1(atoms,G,True)),0.1)
    x1 = x1 + adding
    return x0, x1

class P1GCN(torch.nn.Module):
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer1(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer1(hidden_channels, dataset.num_classes, reduction_type)
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

    
model = P1GCN(hidden_channels = 32, reduction_type = "mean") # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=8e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')

