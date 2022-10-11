from turtle import forward
import torch
import ptens

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
    self.lin = torch.nn.Linear(in_channels, out_channels, bias)
  def forward(self, features: torch.Tensor, graph: ptens.graph):
    return self.lin(ptens.gather(features,graph))