from math import sqrt
from turtle import forward
from typing import Iterator
import torch
import ptens
class Linear(torch.nn.Module):
  def __init__(self,in_channels, out_channels, bias = True) -> None:
    super().__init__()
    self.w = torch.empty(in_channels,out_channels)
    torch.nn.init.uniform_(self.w,-1/sqrt(in_channels),1/sqrt(in_channels))
    if bias:
      self.b = torch.empty(out_channels)
      torch.nn.init.uniform_(self.b,-1/sqrt(in_channels),1/sqrt(in_channels))
    else:
      self.b = None
  def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
    return [self.w] if self.b is None else [self.w, self.b]
  def train(self, mode: bool = True):
    self.w.requires_grad = mode
    if not self.b is None:
      self.b.requires_grad = mode
    return super().train(mode)
  def forward(self,x: ptens.ptensor0) -> ptens.ptensor0:
    #return x * self.w if self.b is None else (x * self.w + self.b)
    # TODO: make this more efficient.
    if self.b is None:
      return x * self.w
    else:
      x = x * self.w
      x = x + ptens.ptensors0.from_matrix(self.b.broadcast_to(len(x),len(self.b)))
      return x
class GCNConv(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, bias: bool = True, add_self_loops: bool = True) -> None:
    super().__init__()
    r"""
    In its current form, this should be equivalent to PyG's GCNConv with the following options:
      - in_channels = in_channels
      - out_channels = out_channels
      - improved = False
      - normalize = False
    """
    self.lin = Linear(in_channels, out_channels, bias)
    self.add_self_loops = add_self_loops
    # [Nxf] * [fxf'] -> NxNxf
    # Nxf -> NxNxNxf -> Mxf
    # A * M + b
  def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
    return self.lin.parameters()
  def train(self, mode: bool = True):
    self.lin.train(mode)
    return super().train(mode)
  def forward(self, features: ptens.ptensor0, graph: ptens.graph):
    propagated_messages = ptens.gather(features,graph)
    if self.add_self_loops:
      propagated_messages += features
    #print(propagated_messages.size())
    #print(propagated_messages.torch().size())
    #print(type(propagated_messages))
    propagated_messages =self.lin(propagated_messages)
    return propagated_messages
    #return self.lin(ptens.gather(features,graph))