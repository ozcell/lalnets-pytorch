import torch as K
import torch.nn.functional as F

class GAR(object):
    def __init__(self, n_p=2, k=5, c_alpha=1, c_beta=1, c_F=1, per_parent=True, symmetric=True):
        super(GAR, self).__init__()

        if per_parent:
            self.k = k 
            self.n_p = n_p            
        else:
            self.k = k * n_p
            self.n_p = 1

        self.c = [c_alpha, c_beta, c_F]
        self.sym = symmetric
        
    def __call__(self, Z):
        Z = Z.view(-1, self.k, self.n_p)
        Bs = [F.relu(Z)]
        if self.sym:
            Bs.append(F.relu(-Z))
            
        affinity = 0
        balance = 0
    
        for B in Bs:

            N = K.bmm(B.permute(2,1,0), B.permute(2,0,1))
            N_sum = N.sum(dim=[1,2])
            N_trace = N.diagonal(dim1=1, dim2=2).sum(-1)

            v = N.diagonal(dim1=1, dim2=2).unsqueeze(1)
            V = K.bmm(v.permute(0,2,1), v.permute(0,1,2))
            V_sum = V.sum(dim=[1,2])
            V_trace = V.diagonal(dim1=1, dim2=2).sum(-1)

            affinity += ((N_sum - N_trace) / ((self.k-1) * N_trace + 1e-8)).sum()
            balance += ((V_sum - V_trace) / ((self.k-1) * V_trace + 1e-8)).sum()

        affinity /= self.n_p
        balance /= self.n_p
        frob = K.pow(Z, 2).mean()

        self.affinity = affinity
        self.balance = balance
        self.frob = frob

        reg = self.c[0] * affinity + self.c[1] * (len(Bs) - balance) + self.c[2] * frob

        return reg

def acolPool(k, n_parent, dtype=K.float32, device='cuda'):
    a = K.eye(n_parent, dtype=dtype, device=device) # np X np 
    for i in range(1, k):
        a = K.cat((a, K.eye(n_parent, dtype=dtype, device=device)),dim=0)
    return a