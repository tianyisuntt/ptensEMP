import torch
import ptens
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
        x2 = ptens.ptensors1.from_matrix(x2,atoms)
        return x2

class Model(torch.nn.Module):
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
        x = self.embedding(x)
        x = ptens.ptensors0.from_matrix(x)
        x = ptens.linmaps1(x)
        x = self.conv1(x,G)
        x = self.conv2(x,G)
        x = self.conv3(x,G)
        x = self.conv4(x,G)
        x = ptens.linmaps0(x,False)
        x = x.torch()
        x = self.pooling(x,batch)
        x = self.batchnorm(x)
        x = self.activ1(self.lin1(x))
        x = self.activ2(self.lin2(x))
        x = self.lin3(x)
        return x
