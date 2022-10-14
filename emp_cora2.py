# Implement the ptens layers on random initializations.

import torch
import ptens as p
from ptens.functions import *
A=p.ptensors1.randn([[1,2,3],[3,5],[2]],3)
print("A=", A)
M=torch.randn(3,5)
print("M=", M)
M_p=p.ptensors0.from_matrix(M)
print("M_p=", M_p)

A=p.ptensors0.sequential([[1],[2],[3]],5)
A_t=A.torch()

A=p.ptensors1.randn([[1,2],[2,3],[3]],3)
B=p.ptensors1.randn([[1,2],[2,3],[3]],3)
C = A+B
print("A+B = C =", C)

D=p.cat(A,B)
print("A.cat(B) = D =", D)

A=p.ptensors1.randn([[1,2],[2,3],[3]],5)
M=torch.randn(5,2)
E=A*M
print("A*M = E =", E)

A=p.ptensors0.randn(3,3)
print(A)
B=p.relu(A,0.1)
print(B)
print("==========================")
A=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32)
G=p.graph.from_matrix(A)
print("G=",G)
print(G.torch())
print("==========================")
G=p.graph.random(8,0.3)
print(G.torch())
G=p.graph.random(8,0.2)
print(G.nhoods(0))
print(G.nhoods(1))
print(G.nhoods(2))
print("==========================")
A = p.ptensors1.randn([[1,2],[3]],3)
G = p.graph.from_matrix(torch.ones(3,2))
B = p.transfer2(A,[[1,2],[2,3]],G) #f?

print("B=", B)
