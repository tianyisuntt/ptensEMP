from math import sqrt
from turtle import forward
import torch
import ptens
class Linear(torch.nn.Module):
  def __init__(self,in_channels, out_channels, bias = True) -> None:
    super().__init__()
    self.w = torch.empty(in_channels,out_channels)
    torch.nn.init.uniform_(self.w,-sqrt(in_channels),sqrt(in_channels))
    if bias:
      self.b = torch.empty(1,out_channels)
      torch.nn.init.uniform_(self.b,-sqrt(in_channels),sqrt(in_channels))
    else:
      self.b = None
  def forward(self,x: ptens.ptensor0) -> ptens.ptensor0:
    return x * self.w if self.b is None else (x * self.w + self.b)
class GCNConv(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
    super().__init__()
    r"""
    In its current form, this should be equivalent to PyG's GCNConv with the following options:
      - in_channels = in_channels
      - out_channels = out_channels
      - improved = False
      - normalize = False
    """
    self.lin = Linear(in_channels, out_channels, bias)
    # [Nxf] * [fxf'] -> NxNxf
    # Nxf -> NxNxNxf -> Mxf
    # A * M + b
  def forward(self, features: ptens.ptensor0, graph: ptens.graph):
    propagated_messages = ptens.gather(features,graph)
    #print(propagated_messages.size())
    #print(propagated_messages.torch().size())
    #print(type(propagated_messages))
    propagated_messages =self.lin(propagated_messages)
    return propagated_messages
    #return self.lin(ptens.gather(features,graph))