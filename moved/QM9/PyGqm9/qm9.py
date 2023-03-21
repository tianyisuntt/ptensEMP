"""QM9Dataset"""
# https://docs.dgl.ai/en/0.8.x/generated/dgl.data.QM9Dataset.html#qm9dataset
#data = QM9Dataset(label_keys=['mu', 'gap'], cutoff=5.0)
#data.num_labels

"""QM9EdgeDataset"""
# https://docs.dgl.ai/en/0.8.x/generated/dgl.data.QM9EdgeDataset.html#dgl.data.QM9EdgeDataset


"""QM9EdgeDataset is better"""
#data = QM9EdgeDataset(label_keys=['mu', 'alpha'])

""" 
    Preprocessing QM9Edge dataset for dgl.data.QM9EdgeDatset.
    When procssing this dataset, we partially refer to the implementation from https://github.com/Jack-XHP/DGL_QM9EDGE  
"""

import os
import numpy as np

from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive, _get_dgl_url
from dgl.convert import graph as dgl_graph
from dgl import backend as F

''' rkdit package for processing moleculars '''
import rdkit
#from rdkit import Chem
#from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import GetBondType as BT
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


HAR2EV = 27.2113825435      # 1 Hartree = 27.2114 eV 
KCALMOL2EV = 0.04336414     # 1 kcal/mol = 0.043363 eV
conversion = F.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


if __name__ == '__main__':

    raw_dir = 'data'
    raw_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    
    keys = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
            'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom',
            'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}

    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    ''' download raw files '''
  #  if not os.path.exists(f'{raw_dir}/gdb9.sdf.csv'):
  #      file_path = download(raw_url, raw_dir)
  #      extract_archive(file_path, raw_dir, overwrite=True)
  #      os.unlink(file_path)

  #  if not os.path.exists(f'{raw_dir}/uncharacterized.txt'):
  #      file_path = download(raw_url2, raw_dir)
  #      os.replace(f'{raw_dir}/3195404', f'{raw_dir}/uncharacterized.txt')


    ''' load raw data '''
    print('loading raw data')
    with open(f'{raw_dir}/gdb9.sdf.csv', 'r') as f:
        target = f.read().split('\n')[1:-1]
        target = [[float(x) for x in line.split(',')[1:20]] for line in target]
        target = F.tensor(target, dtype=F.data_type_dict['float32'])
        target = F.cat([target[:, 3:], target[:, :3]], dim=-1)
        target = (target * conversion.view(1, -1)).tolist()

    with open(f'{raw_dir}/uncharacterized.txt', 'r') as f:
        skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

    suppl = Chem.Suppliers.MolSupplier(f'{raw_dir}/gdb9.sdf', removeHs=False, sanitize=False)

    n_node = []
    n_edge = []
    node_pos = []
    node_attr = []

    src = []
    dst = []
    
    edge_attr = []
    targets = []
    
    ''' process graphs '''
    print('processing graphs')
    for i, mol in enumerate(suppl):
        if i in skip:
            continue
            
        n_atom = mol.GetNumAtoms()

        pos = suppl.GetItemText(i).split('\n')[4:4 + n_atom]
        pos = [[float(x) for x in line.split()[:3]] for line in pos]

        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        
        for atom in mol.GetAtoms():
            type_idx.append(types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

        edge_index = np.array([row, col]).astype(np.int64)
        edge_type = np.array(edge_type).astype(np.int64)
        edge_feat = np.eye(len(bonds))[edge_type]

        perm = (edge_index[0] * n_atom + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_feat = edge_feat[perm]
        
        row, col = edge_index
        hs = (np.array(atomic_number) == 1).astype(np.int64)

        x = F.tensor(hs[row], dtype=F.data_type_dict['float32'])
        idx = F.tensor(col, dtype=F.data_type_dict['int64'])
        num_hs = F.scatter_add(x, idx, n_atom)

        x1 = np.eye(len(types))[type_idx]
        x2 = np.array([atomic_number, aromatic, sp, sp2, sp3, num_hs]).transpose()
        x = np.concatenate((x1,x2), axis = 1)
        
        n_node.append(n_atom)
        n_edge.append(mol.GetNumBonds() * 2)

        node_pos.append(np.array(pos))
        node_attr.append(x)
        
        src += list(row)
        dst += list(col)
        edge_attr.append(edge_feat)   
        targets.append(np.array(target[i]).reshape([1,19]))


    node_attr = np.concatenate(node_attr, axis = 0)
    node_pos = np.concatenate(node_pos, axis = 0)
    edge_attr = np.concatenate(edge_attr, axis = 0)
    targets = np.concatenate(targets, axis = 0)

    n_cumsum = np.concatenate([[0], np.cumsum(n_node)])
    ne_cumsum = np.concatenate([[0], np.cumsum(n_edge)])

    ''' save processed data '''

    np.savez_compressed(f'{raw_dir}/test.npz',
                        n_node=n_node,
                        n_edge=n_edge,
                        node_attr=node_attr,
                        node_pos=node_pos,
                        edge_attr=edge_attr,
                        src=src,
                        dst=dst,  
                        targets=targets)
    print('end')
