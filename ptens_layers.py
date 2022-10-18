from math import sqrt
from typing import Iterator
import torch
import ptens
class Linear(torch.nn.Module):
  def __init__(self,in_channels, out_channels, bias = True) -> None:
    super().__init__()
    r"""
    This follows Glorot initialization for weights.
    """
    self.w = torch.empty(in_channels,out_channels)
    self.b = torch.empty(out_channels) if bias else None
    self.reset_parameters()
  def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
    return [self.w] if self.b is None else [self.w, self.b]
  def reset_parameters(self):
    in_channels = self.w.size(0)
    self.w = torch.nn.init.uniform_(self.w,-1/sqrt(in_channels),1/sqrt(in_channels))
    if not self.b is None:
      self.b = torch.nn.init.zeros_(self.b)
  def train(self, mode: bool = True):
    self.w.requires_grad = mode
    if not self.b is None:
      self.b.requires_grad = mode
    return super().train(mode)
  def forward(self,x: ptens.ptensor0) -> ptens.ptensor0:
    return x * self.w if self.b is None else ptens.linear(x,self.w,self.b) 
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
  def reset_parameters(self):
    self.lin.reset_parameters()
  def train(self, mode: bool = True):
    self.lin.train(mode)
    return super().train(mode)
  def forward(self, features: ptens.ptensors0, graph: ptens.graph):
    assert isinstance(graph,ptens.graph)
    assert isinstance(features,ptens.ptensors0)
    propagated_messages = ptens.gather(features,graph)
    if self.add_self_loops:
      propagated_messages += features
    #print(propagated_messages.size())
    #print(propagated_messages.torch().size())
    #print(type(propagated_messages))
    propagated_messages =self.lin(propagated_messages)
    return propagated_messages
    #return self.lin(ptens.gather(features,graph))