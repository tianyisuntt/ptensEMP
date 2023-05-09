import torch
import ptens
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.transforms.random_node_split import RandomNodeSplit

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0] 
transform_nodes = RandomNodeSplit(split = 'train_rest',
                                  num_val = 433,
                                  num_test = 540)
data = transform_nodes(data)

class GraphAttentionLayer_P0(nn.Module):
    """
    An implementation of GATConv layer in ptens. 
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dropout_prob = 0.5, leakyrelu_alpha = 0.5, relu_alpha = 0.5, concat=True):
        super(GraphAttentionLayer_P0, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
        self.leakyrelu_alpha = leakyrelu_alpha
        self.relu_alpha = relu_alpha
        self.concat = concat

        self.W = ptens.modules.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.W.data)
        self.a = Parameter(torch.empty(out_channels, 1))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = LeakyReLU(self.leakyrelu_alpha)
        self.relu = relu(self.relu_alpha)

    # ptensors0 -> tensor -> do -> ptensors0
    def forward(self, h: ptensors0, adj: ptensors0):
        h_torch = h.torch()
        adj_torch = adj.torch()
        Wh = torch.mm(h_torch, self.W) # h: tensor (N, in_channels); Wh: tensor (N, out_channels)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e) 
        attention = torch.where(adj_torch > 0, e, zero_vec) 
        attention = softmax(attention, dim=1)
        attention = dropout(attention, self.d_prob, training = self.training)
        h_prime = torch.matmul(attention, Wh)
        
        h_prime_p0 = ptens.ptensors0.from_matrix(h_prime)
        if self.concat:
            return relu(h_prime_p0)
        else:
            return h_prime_p0
        
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_channels, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_channels:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
        


class P0GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.gatconv1 = ptens.modules.GraphAttentionLayer_P0(dataset.num_features, hidden_channels)
        self.gatconv2 = ptens.modules.GraphAttentionLayer_P0(hidden_channels, dataset.num_classes)
        self.dropout = ptens.modules.Dropout(prob=0.5)

    def forward(self, x, edge_index):
        x = ptens.ptensors0.from_matrix(x)
        x = self.gatconv1(x)
        x = x.relu()
        x = self.gatconv2(x)
        x = self.conv2(x)
        x = x.torch()
        return x


def train():
      model.train()
      optimizer.zero_grad()  
      out = model(data.x, data.edge_index)
     # print(out)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  
      loss.backward() 
      optimizer.step() 
      return loss
def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum()) 
      return train_acc, test_acc

    
model = P0GAT(hidden_channels = 32) # subject to change
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(1, 201):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("Train accuracy:", train_acc, "Test accuracy:", test_acc)
print("Train Accuracy:", train_acc, ". Test Accuracy:", test_acc, ".")
print('=================================================================')
