import torch 
import utils 
import ntk
from functorch import jacrev, vmap

def get_grads(alphas, train_X, Xs, M, ntk_depth=1):
    
    M_is_passed = M is not None
    sqrtM = None
    if M_is_passed:
        sqrtM = utils.matrix_sqrt(M)
    
    def get_solo_grads(sol, X, x):
        if M_is_passed:
            X_M = X@sqrtM

        def egop_fn(z):
            if M_is_passed:
                z_ = z@sqrtM
            else:
                z_ = z
            K = ntk.relu_ntk(X_M, z_, M=None, depth=ntk_depth)[1]
            return (sol @ K).squeeze()
        grads = vmap(jacrev(egop_fn))(x.unsqueeze(1)).squeeze()
        grads = torch.nan_to_num(grads)
        return grads 

    n, d = train_X.shape
    s = len(Xs)
    
    chunk = 1000
    train_batches = torch.split(torch.arange(n), chunk)

    egop = 0
    sol = jnp.array(alphas.T)
    grads = 0
    for btrain in train_batches:
        grads += get_solo_grads(sol[o][btrain], train_X[btrain], Xs)
    G = G.reshape(-1, d)
    egop += G.T @ G/s

    return egop