# Implement the ptens layers on the Cora dataset.

import torch
import ptens as p
import numpy as np
from deeprobust.graph.data import Dataset
print('=================================================================================')
data = Dataset(root='/tmp/', name='cora', seed=15)
print(data) #cora(adj_shape=(2485, 2485), feature_shape=(2485, 1433))
#print(data.get_prognn_splits())
#print(data.get_train_val_test())
adj, features, labels = data.adj, data.features, data.labels
print('shape of adjacency matrix:', adj.shape) #(2485, 2485)
print('shape of feature matrix:', features.shape) #(2485, 1433)
n_features = features.shape[1]
print('# of features:', n_features) #1433
print('# of labels:', labels.shape) #(2485,)
n_classes = len(np.unique(labels)) 
print('# of classes:', n_classes) #7

adj_matrix = torch.from_numpy(adj.toarray())
feature_matrix = torch.from_numpy(features.toarray())
labels_matrix = torch.tensor(labels)
#print('adj_matrix:', adj_matrix)
#print('feature_matrix:',feature_matrix)
#print('labels_matrix:', labels_matrix)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print('idx_train.shape, idx_val.shape, idx_test.shape: ', idx_train.shape, idx_val.shape, idx_test.shape) #(247,) (249,) (1988,)

print('==================================graph===========================================')
G = p.graph.from_matrix(adj_matrix)
#print('G.nhoods(0):', len(G.nhoods(0)))
#print('G.nhoods(2):', G.nhoods(1))
#print('G.nhoods(3):', G.nhoods(2))
G_t = G.torch() 
print('G_t.shape:', G_t.shape) #torch.Size([2485, 2485])
G_p0 = p.ptensors0.from_matrix(G_t) 
print('G_p0 # of channels:', G_p0.get_nc()) #2485
print('G_p0 # of atoms:', len(G_p0.get_atoms())) #2485
G_p0_p1 = p.linmaps1(G_p0)
print('===G_p0 -> G_p1===')
print('G_p0_p1 # of channels:', G_p0_p1.get_nc()) #2485
print('G_p0_p1 # of atoms:', len(G_p0_p1.get_atoms())) #2485
print('===G_p0 -> G_p2===')
G_p0_p2 = p.linmaps2(G_p0)
print('G_p0_p2 # of channels:', G_p0_p2.get_nc()) #4970
print('G_p0_p2 # of atoms:', len(G_p0_p2.get_atoms())) #2485
print('===G_p1 -> G_p2===')
G_p1_p2 = p.linmaps2(G_p0_p1)
print('G_p1_p2 # of channels:', G_p1_p2.get_nc()) #12425
print('G_p1_p2 # of atoms:', len(G_p1_p2.get_atoms())) #2485
print('===G_p2 -> G_p2===')
G_p2_p2 = p.linmaps2(G_p1_p2)
print('G_p2_p2 # of channels:', G_p2_p2.get_nc()) #186375
print('G_p2_p2 # of atoms:', len(G_p2_p2.get_atoms())) #2485
#print('G_p2_p2[0]:', G_p2_p2[0]) # every single channel is [0]
#print('G_p2_p2[100]:', G_p2_p2[100]) # every single channel is [0]

# TODO: change the channels of a graph to # of classes

print('==================================adj_matrix======================================')
A0 = p.ptensors0.from_matrix(adj_matrix) 
A0_0 = A0[0] 
print('A0 # of channels:', A0.get_nc()) #2485
print('A0 # of atoms:', len(A0.get_atoms())) #2485
U01 = p.unite1(A0, G) 
print('U01 # of channels:', U01.get_nc()) #2485
print('U01 # of atoms:', len(U01.get_atoms())) #2485
U02 =  p.unite2(A0, G) 
print('U02 # of channels:', U02.get_nc()) #4970
print('U02 # of atoms:', len(U02.get_atoms())) #2485
A0_r = p.relu(A0, 0.1)
print('A0_r # of channels:', A0_r.get_nc()) #2485
print('A0_r # of atoms:', len(A0_r.get_atoms())) #2485
B0=p.gather(A0, G)                                              ####### graph 
B0_0 = B0[0] 
print('B0 # of channels:', B0.get_nc()) #2485
print('B0 # of atoms:',len(B0.get_atoms())) #2485

A1 = p.linmaps1(A0) 
print('A1 # of channels:', A1.get_nc()) #2485
print('A1 # of atoms:',len(A1.get_atoms())) #2485
U11=p.unite1(A1, G)
print('U11 # of channels:', U11.get_nc()) #4970
print('U11 # of atoms:', len(U11.get_atoms())) #2485
U12=p.unite2(A1, G)
print('U12 # of channels:', U12.get_nc()) #12425
print('U12 # of atoms:', len(U12.get_atoms())) #2485
#A1_r = p.relu(A1, 0.1) #ok
#print('A1_r # of channels:', A1_r.get_nc()) 
#print('A1_r # of atoms:', len(A1_r.get_atoms())) 

#B1=p.gather(A1, G) #f                   
#print('B1 # of channels:', B1.get_nc())
#print('B1 # of atoms:',len(B1.get_atoms()))

print('===============================U*feature_matrix===================================')
print('feature_matrix.shape:', feature_matrix.shape) # torch.Size([2485, 1433])
transform1 = feature_matrix
A0_f = A0*transform1  # 2485x2485 * 2485x1433                   ###### layer
print('A0_f # of channels:', A0_f.get_nc())  #1433
print('A0_f # of atoms:', len(A0_f.get_atoms())) #2485
#print('f.shape:', f.shape) #torch.Size([1])
#f = B0*feature_matrix #f, since dimensions are different

hidden_channels = 256 # to be determined
transform2 = torch.ones(1433,hidden_channels)   
A0_t2 = A0_f*transform2                                         ###### layer
print('A0_t2 # of channels:', A0_t2.get_nc()) #256
print('A0_t2 # of atoms:', len(A0_t2.get_atoms())) #2485

transform3 = torch.ones(hidden_channels,n_classes)
A0_t3 = A0_t2*transform3
print('A0_t3 # of channels:', A0_t3.get_nc()) #7
print('A0_t3 # of atoms:', len(A0_t3.get_atoms())) #2485
print('===============================feature_matrix=====================================')
G_feature = p.graph.from_matrix(feature_matrix)
G_feature_t = G_feature.torch()
G_f = p.ptensors0.from_matrix(G_feature_t) 
print('G_feature # of channels:', G_f.get_nc()) #1433
print('G_feature # of atoms:', len(G_f.get_atoms())) #2485

A_feature = p.ptensors0.from_matrix(feature_matrix)
print('M_feature # of channels:', A_feature.get_nc()) #1433
print('M_feature # of atoms:', len(A_feature.get_atoms())) #2485

A1_f = p.linmaps1(A_feature)
print('A1_f # of channels:', A1_f.get_nc()) #1433
print('A1_f # of atoms:', len(A1_f.get_atoms())) #2485
A1_f0 = A1_f[0]

#U1_f=p.unite1(A_feature, G_f) #f
#U2_f=p.unite2(A_feature, G_f) #f

#L1_f = p.transfer1(U_f, U_f.get_atoms(), G_f) 
#print('L1_f.shape:', L1_f.shape)
##B_f = p.gather(A_feature, G_f) #f
##print('B_f # of channels:', B_f.get_nc())
##print('B_f # of atoms:',len(B_f.get_atoms()))



