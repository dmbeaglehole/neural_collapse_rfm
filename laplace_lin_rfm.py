import torch
from kernels import laplacian

import utils 
import ntk
from functorch import jacrev, vmap

def get_grads(alphas, train_X, Xs):
    

    def get_solo_grads(sol, X, x):

        def egop_fn(z):
            if M_is_passed:
                z_ = z@sqrtM
            else:
                z_ = z
            K = laplacian(z_, X)
            return (K@sol).squeeze()
        grads = vmap(jacrev(egop_fn))(x.unsqueeze(1)).squeeze()
        grads = torch.nan_to_num(grads)
        return grads 

    n, d = train_X.shape
    s = len(Xs)
    
    chunk = 1000
    train_batches = torch.split(torch.arange(n), chunk)

    egop = 0
    G = 0
    for btrain in train_batches:
        G += get_solo_grads(alphas[btrain,:], train_X[btrain], Xs)
    G = G.reshape(-1, d)
    egop += G.T @ G/s

    return egop

def laplace_lin_rfm(X, y, bandwidth, reg=0., M_batch_size=10000):
    n, d = train_X.shape
    M = None
    
    K = laplacian(X, X, bandwidth)

    alphas = torch.linalg.solve(K, y)

    M = get_grads(alphas, X, X)
        
    return M
