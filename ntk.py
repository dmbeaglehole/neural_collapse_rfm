eps = 1e-10
import torch
from torch import acos, pi, einsum, clip
from torch.nn.functional import cosine_similarity
domain_check = lambda u: clip(u, -1+eps, 1-eps)
kappa_0 = lambda u: (1-acos(domain_check(u))/pi)
kappa_0_ = lambda u: (1/(1-domain_check(u).pow(2)).sqrt()/pi)
kappa_1 = lambda u: (u*kappa_0(u) + (1-domain_check(u).pow(2)).sqrt()/pi)
kappa_1_ = kappa_0
import time

print("Importing")

torch.set_default_dtype(torch.float32)

def norm_M(X,M):
    return (X*(X @ M)).sum(dim=-1).sqrt()

def cosine_similarity_M(X,Z,M):
    nx,dx=X.shape
    nz,dz=Z.shape
    assert dx==dx
    return (X @ M @ Z.T) /norm_M(X,M).view(nx,1)/norm_M(Z,M).view(nz)
    #return einsum('na,mb,ab->nm',X,Z,M)/norm_M(X,M).view(nx,1)/norm_M(Z,M).view(nz)

def ntk_relu(X, Z=None, depth=1, bias=0., M=None):
    """
    Returns the evaluation of nngp and ntk kernels
    for fully connected neural networks
    with ReLU nonlinearity.

    depth  (int): number of weight layers of the network
    bias (float): (default=0.)
    """
    nx, d = X.shape
    if M is None: M = torch.eye(d).to(X.device)
    norm_x = norm_M(X,M).view(nx,1)
    if Z is not None:
        nz, dz = Z.shape
        assert d==dz
        norm_z = norm_M(Z,M).view(nz)
    else:
        nz, norm_z, Z = nx, norm_x.view(nx), X
    S = cosine_similarity_M(X, Z, M)
    Q = S + bias**2/norm_x/norm_z
    for k in range(1, depth):
        Q = Q * kappa_0(S) +  bias**2/norm_x/norm_z
        S = kappa_1(S)
        Q = S + Q
    return S*norm_x*norm_z, Q*norm_x*norm_z