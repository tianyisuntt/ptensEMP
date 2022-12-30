from typing import List, Optional
import torch
import ptens as p
import PtensModules 
import Dropouts

class HeteroLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_types, is_sorted, bias= True) -> None:
        super().__init__()
        self.w = torch.nn.parameter.Parameter(torch.empty(in_channels,out_channels))
        self.b = torch.nn.parameter.Parameter(torch.empty(out_channels)) if bias else None
        self.lins = None
        else:
            self.lins = torch.nn.ModuleList([
                PtensModules.Linear(in_channels, out_channels, bias= True)
                for _ in range(num_types)
            ])
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.w = torch.nn.init.xavier_uniform_(self.w)
        if not self.b is None:
            self.b = torch.nn.init.zeros_(self.b)
 # TODO:           
    def forward(self, x: p.ptensors0, type_vec: p.ptensors0) -> p.ptensors0:
        assert x.get_nc() == self.w.size(0)
        if torch_geometric.typing.WITH_PYG_LIB:
            assert self.w is not None

            perm: Optional[p.ptensors1] = None
            if not self.is_sorted:
                if (type_vec[1:] < type_vec[:-1]).any():
                    type_vec, perm = type_vec.sort()
                    x = x[perm]

            type_vec_ptr = torch.ops.torch_sparse.ind2ptr(
                type_vec, self.num_types)
            out = pyg_lib.ops.segment_matmul(x, type_vec_ptr, self.w)
            if self.b is not None:
                out += self.b[type_vec]

            if perm is not None:  # Restore original order (if necessary).
                out_unsorted = torch.empty_like(out)
                out_unsorted[perm] = out
                out = out_unsorted
        else:
            assert self.lins is not None
            out = p.ptensors0.from_matrix(x.new_empty(x.size(0), self.out_channels))
            for i, lin in enumerate(self.lins):
                mask = type_vec == i
                out[mask] = lin(x[mask])
        return out
    
