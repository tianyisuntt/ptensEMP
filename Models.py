import torch
import ptens
from typing import Callable
from ptens_base import atomspack
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
from torch_geometric.transforms import BaseTransform, Compose
from Transforms import PreComputeNorm
from torch_geometric.transforms import RemoveIsolatedNodes

class CleanupData(BaseTransform):
    def __call__(self, data):
        data.x = data.x
        data.y = data.y.float()
        return data
      
class ToPtens_Batch(BaseTransform):
    def __call__(self, data):
        data.G = ptens.graph.from_matrix(torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1),dtype=torch.float32),
                                                             size=(data.num_nodes,data.num_nodes)).to_dense())
        return data
      
class ConvolutionalLayer(torch.nn.Module):
    def __init__(self,channels_in: int,channels_out: int,nhops: int) -> None:
        super().__init__()
        self.batchnorm1 = torch.nn.BatchNorm1d(channels_in * 2)
        self.lin1 = torch.nn.Linear(channels_in * 2,channels_in)
        self.activ1 = torch.nn.ReLU(True)
        self.batchnorm2 = torch.nn.BatchNorm1d(channels_in )
        self.lin2 = torch.nn.Linear(channels_in ,channels_out)
        self.activ2 = torch.nn.ReLU(True)
        self.nhops = nhops
    def forward(self, x: ptens.ptensors1, G: ptens.graph) -> ptens.ptensors1:
        x1 = x.transfer1(G.nhoods(self.nhops),G,False)
        atoms = x1.get_atoms()
        x2 = self.batchnorm1(x1.torch())
        x2 = self.activ1(self.lin1(x2))
        x2 = self.batchnorm2(x2)
        x2 = self.activ2(self.lin2(x2))
        x3 = ptens.ptensors1.from_matrix(x2,atoms)
        return x3

class GPP(torch.nn.Module):
    # Graph property prediction
    def __init__(self, embedding_dim: int, convolution_dim: int, dense_dim: int,
                 pooling: Callable[[torch.Tensor,torch.Tensor],torch.Tensor]) -> None:
        super().__init__()
        self.embedding = torch.nn.Linear(11,embedding_dim)
        self.conv1 = ConvolutionalLayer(embedding_dim,   convolution_dim, 1)
        self.conv2 = ConvolutionalLayer(convolution_dim, convolution_dim, 2)
        self.conv3 = ConvolutionalLayer(convolution_dim, convolution_dim, 3)
        self.conv4 = ConvolutionalLayer(convolution_dim, dense_dim,       4)
        self.pooling = pooling
        self.batchnorm = torch.nn.BatchNorm1d(dense_dim)
        self.lin1 = torch.nn.Linear(dense_dim,dense_dim)
        self.activ1 = torch.nn.ReLU(True)
        self.lin2 = torch.nn.Linear(dense_dim,dense_dim)
        self.activ2 = torch.nn.ReLU(True)
        self.lin3 = torch.nn.Linear(dense_dim,dense_dim)
    def forward(self, x: torch.Tensor, G: ptens.graph, batch: torch.Tensor) -> torch.Tensor:
    #  QM9
        x = self.embedding(x)
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x)
        x = self.conv1(x,G)
        x = self.conv2(x,G)
        x = self.conv3(x,G)
        x = self.conv4(x,G)
        x = ptens.linmaps0(x,False)
        x = self.pooling(x.torch(),batch)
        x = self.batchnorm(x)
        x = self.activ1(self.lin1(x))
        x = self.activ2(self.lin2(x))
        x = self.lin3(x)
        return x


class P1GCN(torch.nn.Module):
    # First order graph neural network
    def __init__(self, hidden_channels, reduction_type):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_1P(dataset.num_features, hidden_channels, reduction_type)
        self.conv2 = ptens.modules.ConvolutionalLayer_1P(hidden_channels, dataset.num_classes, reduction_type)
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)
    def forward(self, x, edge_index, normalization):
    # normalization = False: Cora, Cornell, CiteSeer, PubMed, Wisconsin
    # normalization = True: (AmazonPhoto)
        x = ptens.linmaps1(x, False)
        x = self.conv1(x,edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = ptens.linmaps0(x, normalization).torch()
        x = F.log_softmax(x, dim=1)
        x = ptens.ptensors0.from_matrix(x)
        return x


class P0GCN(torch.nn.Module):
    # Zeroth order graph neural network
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = ptens.modules.ConvolutionalLayer_0P(dataset.num_features, hidden_channels)
        self.conv2 = ptens.modules.ConvolutionalLayer_0P(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)
    def forward(self, x, edge_index):
    # Wisconsin, PubMed, Cornell, Cora, CiteSeer, (AmazonPhoto)
        x = self.conv1(x,edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class PMLP(torch.nn.Module):
    # Zeroth order multilayer perceptron
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.plin1 = ptens.modules.Linear(dataset.num_features, hidden_channels)
        self.plin2 = ptens.modules.Linear(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5, device = None)
    def forward(self, x, edge_index):
    # CiteSeer, Cora, PubMed, Wisconsin, (AmazonPhoto)
        x = ptens.ptensors0.from_matrix(x)
        x = self.plin1(x).relu()
        x = self.dropout(x)
        x = self.plin2(x).torch()
        return x


class PMLP1(torch.nn.Module):
    # Zeroth order multilayer perceptron version1
    def __init__(self, hidden_channels1, hidden_channels2, hidden_channels3):
        super().__init__()
        torch.manual_seed(12345)
        self.plin1 = ptens.modules.Linear(dataset.num_features, hidden_channels1)
        self.plin2 = ptens.modules.Linear(hidden_channels1, hidden_channels2)
        self.plin3 = ptens.modules.Linear(hidden_channels2, hidden_channels3)
        self.plin4 = ptens.modules.Linear(hidden_channels3, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.6, device = None)
    def forward(self, x, edge_index):
    # Cornell
        x = ptens.ptensors0.from_matrix(x)
        x = self.plin1(x).relu()
        x = self.dropout(x)
        x = self.plin2(x).relu()
        x = self.dropout(x)
        x = self.plin3(x).relu()
        x = self.dropout(x)
        x = self.plin4(x).torch()
        return x


class SubGraph(torch.nn.Module):
    def __init__(self, hidden_channels, subgraph):
        super().__init__()
        torch.manual_seed(12345)
        self.uniteconv1 = ptens.modules.LazyUnite(hidden_channels)
        self.batchnorm1 = ptens.modules.LazyBatchNorm()
        self.dropout1 = ptens.modules.Dropout(prob=0.5, device = None)
        self.graphconv1 = ptens.modules.LazyHigherGraphConv(hidden_channels)
        self.batchnorm2 = ptens.modules.LazyBatchNorm()
        self.dropout2 = ptens.modules.Dropout(prob=0.5, device = None)
        self.transferconv1 = ptens.modules.LazyTransfer(hidden_channels)
        self.batchnorm2 = ptens.modules.LazyBatchNorm()
        self.dropout3 = ptens.modules.Dropout(prob=0.5, device = None)
        self.linear1 = ptens.modules.LazyLinear(hidden_channels)
        self.batchnorm3 = ptens.modules.LazyBatchNorm()
        self.dropout4 = ptens.modules.Dropout(prob=0.5, device = None)       
    def forward(self, x, edge_index):
    # input is a torch tensor
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x, False)
        x = self.uniteconv1(x)
        x = self.batchnorm1(x).relu()
        x = self.dropout1(x)
        x = self.graphconv1(x)
        x = self.batchnorm2(x).relu()
        x = self.dropout2(x)
        x = self.transferconv1(x)
        x = self.batchnorm2(x).relu()
        x = self.dropout3(x)
        x = self.linear1(x).relu()
        x = self.batchnorm3(x).relu()
        x = self.dropout4(x)
        return x

class GraphTransfer(torch.nn.Module):
    # subgraph transfer learning
    def __init__(self, in_channels, out_channels, hidden_channels, subgraph):
      super().__init__()
      torch.manual_seed(12345)
      self.uniteconv1 = ptens.modules.LazyUnite(in_channels,1)
      self.batchnorm1 = ptens.modules.LazyBatchNorm()
      self.dropout1 = ptens.modules.Dropout(prob=0.5, device = None)
      self.graphconv1 = ptens.modules.LazyHigherGraphConv(hidden_channels)
      self.batchnorm2 = ptens.modules.LazyBatchNorm()
      self.dropout2 = ptens.modules.Dropout(prob=0.5, device = None)
      self.graphconv2 = ptens.modules.LazyHigherGraphConv(hidden_channels)
      self.batchnorm3 = ptens.modules.LazyBatchNorm()
      self.dropout3 = ptens.modules.Dropout(prob=0.5, device = None)
      self.subgraphconv = SubGraph(hidden_channels, subgraph)
      self.lin1 = torch.nn.Linear(2*hidden_channels,hidden_channels)
      self.activ1 = torch.nn.ReLU(True)
      self.dropout4 = torch.nn.Dropout(p=0.5)
      self.lin2 = torch.nn.Linear(hidden_channels,hidden_channels)
      self.activ2 = torch.nn.ReLU(True)
      self.dropout5 = torch.nn.Dropout(p=0.5)
      self.lin3 = torch.nn.Linear(hidden_channels,1)
    def forward(self, x, edge_index):
    # QM9, ZINC, OGBs
      x = ptens.ptensors0.from_matrix(x)
      x = ptens.linmaps1(x, False)
      x = self.uniteconv1(x)
      x = self.batchnorm1(x).relu()
      x = self.dropout1(x)
      x = self.graphconv1(x)
      x = self.batchnorm2(x).relu()
      x = self.dropout2(x)
      x = self.graphconv2(x)
      x = self.batchnorm3(x).relu()
      x = self.dropout3(x)
      x = self.subgraphconv(x)
      x = ptens.linmaps0(x, False).torch()
      x = global_mean_pool(x)
      x = self.lin1(x).relu()
      x = self.dropout4(x)
      x = self.lin2(x).relu()
      x = self.dropout5(x)
      x = self.lin3(x)
      return x 
      


