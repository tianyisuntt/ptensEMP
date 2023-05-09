from torch_geometric.datasets import QM9

#dp = QM9(root='./data/QM9/').to_datapipe()
#dp = dp.batch_graphs(batch_size=2, drop_last=True)
#print(dp)
#for batch in dp:
#    pass
#from schnetpack import AtomsData
from schnetpack.datasets import QM9
#qm9 = QM9('./qm9.db', batch_size = 500)
#print(qm9)
#at = qm9.QM9.get_atoms(idx = 0)
#print(len(qm9))
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
dataset = QM9('Datasets',pre_transform=on_process_transform) 
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit
class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data
    
class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index
        return data
    
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
#print('path:', osp.dirname(osp.realpath(__file__)))

dataset = QM9(root = './data/QM9/raw/')
print(dataset)
data = dataset[0]
#path = '/Users/tianyisun/desktop/pt/model/data/QM9/raw/'
#dataset = QM9(osp.join(path))
print(data)
transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
#dataset = QM9(path, transform=transform)#.shuffle()


# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

# Split datasets.
test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


#dp = QM9(root='./data/QM9/').to_datapipe()
#dp = dp.batch_graphs(batch_size=2, drop_last=True)

#for batch in dp:
#    pass
