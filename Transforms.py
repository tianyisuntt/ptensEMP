from typing import List, Optional
import torch
import ptens as p
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class PreComputeNorm(BaseTransform):
  def __call__(self, data):
    norm = torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1))).to_dense().sum(0)**-0.5
    #norm = norm.cuda()
    data.norm = norm
    return data
  
class ToPtens_Batch(BaseTransform):
  def __call__(self, data: Data) -> Data:
    #data.G = p.graph.from_edge_index(data.edge_index.float())
    data.G = p.graph.from_matrix(torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1),dtype=torch.float32),size=(data.num_nodes,data.num_nodes)).to_dense())
   # data.norm = p.ptensors0.from_matrix(data.norm.unsqueeze(1))
    return data

class precompute_edge_attr_ptensors(BaseTransform):
  def __call__(self, data: Data) -> Data:
    # this gives us a neighborhood mask for every neighborhood
    adj = torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.size(1),dtype=bool),size=(data.num_nodes,data.num_nodes)).to_dense()
    atom_list = torch.arange(data.num_nodes,dtype=int)
    edge_atoms = [atom_list[adj[i]].tolist() for i in range(data.num_nodes)]
    adj_weighted = torch.sparse_coo_tensor(data.edge_index,data.edge_attr,size=(data.num_nodes,data.num_nodes,data.num_edge_features)).to_dense()
    edge_weights = torch.cat([adj_weighted[i,adj[i]] for i in range(data.num_nodes)]).cuda()
    data.ptens_edge_atoms = edge_atoms
    data.ptens_edge_weights = edge_weights
    return data
"""
class compute_edge_attr_ptensors(BaseTransform):
  def __call__(self, data: Data) -> Data:
    data.edge_ptensors1 = p.ptensors1.from_matrix(data.ptens_edge_atoms,data.ptens_edge_weights)
    return data
"""
