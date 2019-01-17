import torch as K
import torch.nn.functional as F

def GAR(Z, k, n_p, c_alpha=1, c_beta=1, c_F=1, per_parent=True, symmetric=True):
    
    if not per_parent:
        k = k * n_p
        n_p = 1
        
    Z = Z.view(-1, k, n_p)
    Bs = [F.relu(Z)]
    if symmetric:
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

        affinity += ((N_sum - N_trace) / ((k-1) * N_trace + 1e-8)).sum()
        balance += ((V_sum - V_trace) / ((k-1) * V_trace + 1e-8)).sum()

    affinity /= n_p
    balance /= n_p
    frob = K.pow(Z, 2).mean()

    reg = c_alpha * affinity + c_beta * (len(Bs) - balance) + c_F * frob

    return reg