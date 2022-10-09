# Model under construction.

import torch
import ptens as p

class EMP1(torch.nn.Module):
    def __init__(self,x,G):
        super().__init__()
        torch.manual_seed(12345)

        self.pu1 = p.unite1(x,G)
        self.pu2 = p.unite2(x,G)
        self.pl1 = x*feature_matrix
        self.pl2 = x*torch.ones(x.shape[1],n_classes)

    def forward(self,x, G):
        x = self.pu1(x,G)
        x = x.relu(x, 0.1)
        x = self.pu2(x,G)
        return x

x =
y =
M = 
atoms =
nc = 
prelayer =
atoms_thislayer =
G = 
n_vertices = 
prob_e =

class EMP2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        
        self.lms0 = p.linmaps0(x)
        self.lms1 = p.linmaps1(x)
        self.lms2 = p.linmaps2(x)
        
        # tensors tensors transfer
        self.trans0 = p.transfer0(x, atoms)
        self.trans1 = p.transfer1(x, atoms)
        self.trans2 = p.transfer2(x, atoms)
        
        # layer layer transfer
        self.trans0 = p.transfer0(prelayer, atoms_thislayer, G) #f: 0->0; 2->0
        self.trans1 = p.transfer1(prelayer, atoms_thislayer, G) #f: 0->1; 2->1
        self.trans2 = p.transfer2(prelayer, atoms_thislayer, G) #f: 0->2; 2->2

        self.atoms = x.get_atoms()
        self.nc = x.get_nc()

        self.init_rpts0 = p.ptensors0.randn(atoms, nc)
        self.init_seqpts0 = p.ptensors0.sequential(atoms, nc)
        self.init_zpts0 = p.ptensors0.zeros(atoms, nc)
        self.init_opts0 = p.ptensors0.ones(atoms, nc)

        self.init_rpts1 = p.ptensors1.randn(atoms, nc)
        self.init_seqpts1 = p.ptensors1.sequential(atoms, nc)
        self.init_zpts1 = p.ptensors1.zeros(atoms, nc)
        self.init_opts1 = p.ptensors1.ones(atoms, nc)

        self.init_rpts2 = p.ptensors2.randn(atoms, nc)
        self.init_seqpts2 = p.ptensors2.sequential(atoms, nc)
        self.init_zpts2 = p.ptensors2.zeros(atoms, nc)
        self.init_opts2 = p.ptensors2.ones(atoms, nc)
        self.init_rgraph = p.graph.random(n_vertices, prob_e)

        self.init_rtorch = torch.randn(row,col)
        self.init_ztorch = torch.zeros(row,col)
        self.init_otorch = torch.ones(row,col)
        
        self.add = x+y # x and y are ptensors with same dim, atoms, nc.
        self.multi = x*M # M is a torch matrix with x.get_nc() == M.shape[0].
        
        self.concat = p.cat(x, y) # nc and atoms of x and y are same.
        self.relu = p.relu(x, alpha = 0.5) # x is ptensors of any dim.
        self.inp = p.inp(x, y) 
        self.transform = x*M # M = feature_matrix
        
        self.ptens0_to_torch = x.torch()
        self.torch_to_ptens0 = p.ptensors0.from_matrix(x) # ok: pts0

        self.u1 = p.unite1(x,G) # ok: pts0, pts1, pts2
        self.u2 = p.unite2(x,G) # ok: pts0, pts1, pts2
        
        self.gather = p.gather(x,G) # ok: pts0

        self.init_G = p.graph.from_matrix(adj_matrix)
        self.init_G_t = G.torch()        
        self.init_G_adj_p = p.ptensors0.from_matrix(G_t)
        
        self.layer0 = p.ptensors0.from_matrix(adj_matrix)
        self.layer1 = adj_ptensors*feature_matrix
        self.layer2 = f_ptensors*torch.ones(f_ptensors.get_nc(), hidden_channels)
        self.layer3 = h_ptensors*torch.ones(h_ptensors.get_nc(), n_classes)

    def forward(self, adj_matrix, feature_matrix):
        G = p.graph.from_matrix(adj_matrix)
        G_adj_t = G.torch()
        G_adj_p = p.ptensors0.from_matrix(G_adj_t)
        
        x = G_adj_p*feature_matrix
        x = x*torch.ones(x.get_nc(), hidden_channels)
        x = x.relu(x, 0.5)
        x =  x*torch.ones(x.get_nc(), n_classes)
        x = x.relu(x, 0.5)
        return x

print('EMP2 on Cora=======================================================')
from deeprobust.graph.data import Dataset
data = Dataset(root='/tmp/', name='cora', seed=15)
adj, features, labels = data.adj, data.features, data.labels
adj_matrix = torch.from_numpy(adj.toarray())
feature_matrix = torch.from_numpy(features.toarray())
labels_matrix = torch.tensor(labels)
n_classes = len(np.unique(labels)) 

model = EMP2(hidden_channels = 16) # init
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  
      out = model(_,_) # forward
      loss = criterion(out[idx_train], labels[idx_train])  
      loss.backward()  
      optimizer.step()  
      return loss

def test():
      model.eval()
      out = model(_,_)
      pred = out.argmax(dim=1)  
      train_correct = pred[idx_train] == labels[idx_train]  
      train_acc = int(train_correct.sum()) / int(idx_train.sum())  
      test_correct = pred[idx_test] == labels[idx_test]  
      test_acc = int(test_correct.sum()) / int(idx_test.sum())  
      return train_acc, test_acc

for epoch in range(1, 2):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

train_acc, test_acc = test()
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")

