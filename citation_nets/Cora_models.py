import torch
import ptens
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


class PMLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.plin1 = ptens.modules.Linear(dataset.num_features, hidden_channels)
        self.plin2 = ptens.modules.Linear(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5)

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
        x = ptens.linmaps1(x, False)
        x = self.conv1(x,edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = ptens.linmaps0(x, False)
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
        x = self.conv1(x,edge_index).relu()
        x = ptens.linmaps0(x, False).torch()
        x = softmax(x, dim = 1)
        x = ptens.ptensors0.from_matrix(x)
        x = self.dropout(x)  

        x = ptens.linmaps1(x, False)  
        x = self.conv2(x,edge_index).relu()
        x = ptens.linmaps0(x, False).torch()
        x = softmax(x, dim = 1)
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x, False) 
        x = self.dropout(x) 
        
        x = ptens.linmaps0(x, False)
        return x

class P1GCN1(torch.nn.Module):
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
        x = torch.sum(x, 0)
        x = F.log_softmax(x, dim = 1)
        x = ptens.ptensors0.from_matrix(x)
        return x


class ConvolutionalLayer(torch.nn.Module):
    def __init__(self,channels_in: int,channels_out: int, nhops: int) -> None:
        super().__init__()
        self.nhops = nhops
        self.lin1 = ptens.modules.Linear(channels_in * 2,channels_in)
        self.dropout = ptens.modules.Dropout(prob=0.5,device = None)
        self.lin2 = ptens.modules.Linear(channels_in, channels_out)
    def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
        x1 = x.transfer1(G.nhoods(self.nhops), G, False)
        x1 = self.lin1(x1).relu()
        x1 = self.dropout(x1)
        x1 = self.lin2(x1)
        return x1
class PCONV(torch.nn.Module):
    def __init__(self, channels_in: int, convolution_dim: int, dense_dim: int,
                 out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvolutionalLayer(channels_in,   convolution_dim, 1)
        self.conv2 = ConvolutionalLayer(convolution_dim, dense_dim,     4)
        self.lin1 = ptens.modules.Linear(dense_dim,dense_dim)
        self.drop = ptens.modules.Dropout(prob=0.5,device = None)
        self.lin2 = ptens.modules.Linear(dense_dim,out_channels)
    def forward(self, x: torch.Tensor, G: ptens.graph, batch: torch.Tensor) -> torch.Tensor:
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x)
        x = self.conv1(x,G)
        x = self.conv2(x,G) 
        x = self.lin1(x).relu()
        x = self.drop(x)
        x = self.lin2(x)
        x = ptens.linmaps0(x,False)
        return x

