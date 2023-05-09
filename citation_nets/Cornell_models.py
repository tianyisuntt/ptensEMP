import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import WebKB
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
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

class PMLP(torch.nn.Module):
    def __init__(self, hidden_channels1, hidden_channels2, hidden_channels3):
        super().__init__()
        torch.manual_seed(12345)
        self.plin1 = ptens.modules.Linear(dataset.num_features, hidden_channels1)
        self.plin2 = ptens.modules.Linear(hidden_channels1, hidden_channels2)
        self.plin3 = ptens.modules.Linear(hidden_channels2, hidden_channels3)
        self.plin4 = ptens.modules.Linear(hidden_channels3, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.6, device = None)

    def forward(self, x, edge_index):
        x = ptens.ptensors0.from_matrix(x)
        x = self.plin1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.plin2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.plin3(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.plin4(x)
        x = x.torch()
        return x
    
class P0GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_0P(dataset.num_features, hidden_channels)
        self.conv2 = ptens.modules.ConvolutionalLayer_0P(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.6, device = None)

    def forward(self, x, edge_index):
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
class P1GCN(torch.nn.Module):
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)

    def forward(self, x, edge_index, batch):
        x = ptens.linmaps1(x, False)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = ptens.linmaps0(x, False).torch()
        x = F.log_softmax(x, dim = 1)
        x = ptens.ptensors0.from_matrix(x)
        return x

class P1GCN0(torch.nn.Module):
    def __init__(self, hidden_channels1, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels1, reduction_type)
        self.conv4 = ptens.modules.ConvolutionalLayer_1P(hidden_channels1, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.6,device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x, False)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = ptens.linmaps0(x, False)
        return x

class P1GCN2(torch.nn.Module):
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



    
