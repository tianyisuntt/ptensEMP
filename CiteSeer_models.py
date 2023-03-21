import torch
import ptens
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from Transforms import ToPtens_Batch
dataset = Planetoid(root='data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
data = dataset[0]  
transform_nodes = RandomNodeSplit(split = 'test_rest', 
                                  num_train_per_class = 510,
                                  num_val = 500)
data = transform_nodes(data)
on_learn_transform = ToPtens_Batch()
data = on_learn_transform(data)


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


class P0GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_0P(dataset.num_features, hidden_channels)
        self.conv2 = ptens.modules.ConvolutionalLayer_0P(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)

    def forward(self, x, edge_index):       
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class P1GCN0(torch.nn.Module):
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)

    def forward(self, x, edge_index):
        x = ptens.linmaps1(x)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = ptens.linmaps0(x)
        return x
    
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
        x = ptens.linmaps0(x, False).torch()#.sum(axis = 1)
        x = F.log_softmax(x, dim=1)
        x = ptens.ptensors0.from_matrix(x)
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
        x = ptens.linmaps1(x)
        x = self.lin1(x).relu()
        x = self.dropout(x)
        x = self.conv1(x,edge_index)
        x = self.conv2(x,edge_index)
        x = self.conv3(x,edge_index)
        x = self.batchnorm(x)
        x = self.lin2(x).relu()
        x = self.dropout(x)
        x = self.lin3(x)
        x = ptens.linmaps0(x)
        return x
