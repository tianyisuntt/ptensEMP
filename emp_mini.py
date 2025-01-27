# A simple model constructed with the ptens layers.

import torch
import ptens as p
import numpy as np
from deeprobust.graph.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() and not use_ptens else 'cpu')

data = Dataset(root='/tmp/', name='cora', seed=15)
adj, features, labels = data.adj, data.features, data.labels
adj_matrix = torch.from_numpy(adj.toarray())
feature_matrix = torch.from_numpy(features.toarray())
labels_matrix = torch.tensor(labels.T)
n_classes = len(np.unique(labels))
hidden_channels = 16

G = p.graph.from_matrix(adj_matrix)
G_adj_t = G.torch()
G_adj_p = p.ptensors0.from_matrix(G_adj_t).to(device)
        
x = G_adj_p*feature_matrix
x = x*torch.ones(x.get_nc(), hidden_channels)
x = p.relu(x, 0.1)
x = x*torch.ones(x.get_nc(), n_classes)
x = p.relu(x, 0.1)
x = x*torch.ones(x.get_nc(), 1)
x = x.torch().int()
x = torch.reshape(x, (-1,))
print('x=', x)
print('labels=', labels_matrix)

correct = labels_matrix == x
#print(correct)
acc = int(correct.sum()) / int(len(labels))  
print('accuracy: ', acc)
