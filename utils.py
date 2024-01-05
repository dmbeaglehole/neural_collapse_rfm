import torch 

def matrix_sqrt(M, thresh=False):
    S, V = torch.linalg.eigh(M)
    S[S<0] = 0

    if thresh:
        # k = int(3*len(S)//4)
        k = -5
        S[:k] = 0

    S = torch.diag(S**0.5)
    return V @ S @ V.T
