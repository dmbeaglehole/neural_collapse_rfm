import torch
import numpy as np

import rfm
import rff
import rff_laplace

import torch.nn as nn


import os
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

import utils
from functorch import jacrev, vmap

import math


import random

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=500)
parser.add_argument('-dataset', default="cifar")
parser.add_argument('-depth', default=6)
parser.add_argument('-kernel', default="laplace")
parser.add_argument('-bw', default=2.0)
parser.add_argument('-width', default=512)
parser.add_argument('-sigma', default=0.5)
parser.add_argument('-use_rff', action='store_true')
parser.add_argument('-seed', type=int, default=0)
args = parser.parse_args()

for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/"

SEED = int(args.seed)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

DEPTH = int(args.depth)
KERNEL = args.kernel
dataset = args.dataset
n = int(args.n)
BW = float(args.bw)
WIDTH = int(args.width)
SIGMA = float(args.sigma)

use_rff = args.use_rff

def mat_cov(A, B):
    A_ = A.reshape(-1).clone()
    B_ = B.reshape(-1).clone().to(A_.device)
    
    A_ -= A_.mean()
    B_ -= B_.mean()
    
    norm1 = A_.norm()
    norm2 = B_.norm()
    
    return (torch.dot(A_, B_) / norm1 / norm2).item()

def one_hot_data(dataset, num_classes, num_samples):
    Xs = []
    ys = []

    for ix in range(min(len(dataset),num_samples)):
        X,y = dataset[ix]
        Xs.append(X)

        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        ys.append(ohe_y)

    return torch.stack(Xs), torch.stack(ys)

def get_binary(dataset, classes):
    c1, c2 = classes
    
    binary_dataset = []
    for ix in tqdm(range(len(dataset))):
        X,y = dataset[ix]
        
        if y==c1:
            binary_dataset.append((X,0))
        elif y==c2:
            binary_dataset.append((X,1))

    return binary_dataset

def preprocess(X):
    """
    X : (n, c, P, Q)
    """
    Xpp = X - X.mean(dim=0)
    Xpp /= X.std(dim=0).unsqueeze(0)
    return torch.nan_to_num(Xpp)

def get_NC2_from_means(u_, center=True):
    if center:
        u = u_ - u_.mean(dim=0).unsqueeze(0)
    else:
        u = u_.clone()
    u = u / u.norm(dim=1).unsqueeze(1)
    uu = u@u.T
    simplex = torch.eye(NUM_CLASSES)*(1 + 1/(NUM_CLASSES - 1)) - torch.ones((NUM_CLASSES,NUM_CLASSES)) / (NUM_CLASSES - 1)
    simplex = simplex.to(u.device).to(u.dtype)
    return ((uu - simplex)**2).mean().item()

def matrix_sqrt(M):
    M = (M.cpu().numpy()).astype(np.float64)
    S, V = np.linalg.eigh(M)
    S[S<0] = 0
    V = torch.from_numpy(V).float()
    S = torch.from_numpy(S).float()
    S = torch.diag(S**0.5)
    return V @ S @ V.T
    
class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
    def forward(self, x):
        return nn.ReLU()(x)
    
def sample_feats(M, d, k):
    weight_fn = nn.Linear(d, k, bias=False).to(M.device)
    weight_fn.weight = nn.Parameter(weight_fn.weight@M, requires_grad=False)

    return nn.Sequential(weight_fn, 
                      Activation()
                     )

def get_class_variance(X, y):
    
    embeddings = [[] for _ in range(NUM_CLASSES)]
    for x, y in zip(X, y):
        c = torch.argmax(y)
        embeddings[c].append(x)
        
    embeddings = [torch.stack(Xc) for Xc in embeddings]
    class_counts = [len(embeddings[c]) for c in range(NUM_CLASSES)]
    class_means = torch.stack([Xc.mean(dim=0) for Xc in embeddings])
    
    
    muG = class_means.mean(dim=0).unsqueeze(0)
    SigmaB = (class_means - muG).T @ (class_means - muG)
    
    SigmaW = 0.
    SigmaT = 0.
    
    for c in range(NUM_CLASSES):
        muC = class_means[c].unsqueeze(0)
        Xc = embeddings[c]
        
        SigmaWc = (Xc - muC).T @ (Xc - muC)
        SigmaW += SigmaWc
        
        SigmaTc = (Xc - muG).T @ (Xc - muG)
        SigmaT += SigmaTc
        
    SigmaB /= NUM_CLASSES
    SigmaW /= (NUM_CLASSES * n)
    SigmaT /= (NUM_CLASSES * n)

    return SigmaB, SigmaW, SigmaT, class_means

### MAIN CODE ###

if dataset=='mnist':
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "MNIST/"
    trainset = torchvision.datasets.MNIST(root=path,
                                        train = True,
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.MNIST(root=path,
                                        train = False,
                                        transform=transform,
                                        download=True)
elif dataset=='cifar':
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "cifar10/"

    trainset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root=path,
                                           train=False,
                                           transform=transform,
                                           download=True)
    
elif dataset=='svhn':
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "svhn/"

    trainset = torchvision.datasets.SVHN(root=path,
                                         split='train',
                                         transform=transform,
                                         download=True)

    testset = torchvision.datasets.SVHN(root=path,
                                        split='test',
                                        transform=transform,
                                        download=True)

NUM_CLASSES = 10


train_X_, train_y_ = one_hot_data(trainset, NUM_CLASSES, num_samples=n)
test_X_, test_y_ = one_hot_data(testset, NUM_CLASSES, num_samples=100)


## reorder data
newXs = []
newYs = []
for c in range(NUM_CLASSES):
    class1_idx = train_y_[:,c] == 1
    newYs.append(train_y_[class1_idx])
    newXs.append(train_X_[class1_idx])

train_X_ = torch.concat(newXs,dim=0)
train_y_ = torch.concat(newYs,dim=0)
print(train_X_.shape, train_y_.shape)


## -1,+1 labels
train_y_ -= 0.5
train_y_ *= 2

test_y_ -= 0.5
test_y_ *= 2


train_X = train_X_.reshape(len(train_X_),-1)#.double()
test_X = test_X_.reshape(len(test_X_),-1)#.double()

    
train_y = train_y_.to(train_X_.dtype)
test_y = test_y_.to(test_X_.dtype)


train_X = preprocess(train_X)
test_X = preprocess(test_X)



## Run DeepRFM ##

dir_path = f'figures/{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_deeprfm'


def batch_multiply(X):
    mb_size = 25000
    batches = torch.split(torch.arange(len(X)), mb_size)
    xXs = []
    for b1 in batches:
        Xb1 = X[b1].clone().cuda()
        blocks = []
        for b2 in batches:
            Xb2 = X[b2].clone().cuda()
            blocks.append((Xb1@Xb2.T).cpu())
            Xb2 = Xb2.cpu()
        Xb1 = Xb1.cpu()
        row = torch.concat(blocks,dim=1)
        xXs.append(row)
    return torch.concat(xXs, dim=0)

# fig, ax = plt.subplots(1, 1)
# X = train_X.reshape(len(train_X),-1).cpu()
# with torch.no_grad():
#     X = X - X.mean(dim=0).unsqueeze(0)
#     X /= X.norm(dim=1).unsqueeze(1)
#     XXt = batch_multiply(X)

# im = ax.imshow(XXt)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_title(f'Input Data', fontsize=20)
# plt.savefig(os.path.join(dir_path, f'deep_rfm_vis_{dataset}_rff_{use_rff}_sigma_{SIGMA}_n_{n}_layer_0.pdf'), format='pdf')

# exit()

Xs = []
MXs = []
PhiMXs = []

train_X_deep = train_X.reshape(len(train_X),-1).cpu()
test_X_deep = test_X.reshape(len(test_X),-1).cpu()
train_y = train_y.to(train_X_deep.dtype).cuda()
test_y = test_y.to(test_X_deep.dtype).cuda()

del train_X, test_X

def print_total_free_memory():
    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    # print("Total free memory:",r-a)
    a, b = torch.cuda.mem_get_info()
    print(a//10**9, b//10**9)
    return

for dep in range(DEPTH):
    print(f'Depth {dep} of DeepRFM')
    
    
    train_X_deep /= train_X_deep.norm(dim=-1).unsqueeze(1)
    test_X_deep /= test_X_deep.norm(dim=-1).unsqueeze(1)
    
    train_X_deep = train_X_deep.cpu()
    test_X_deep = test_X_deep.cpu()
    

    Xs.append(train_X_deep.cpu().clone())
    
    
    if KERNEL == 'laplace':
        model = rfm.LaplaceRFM(bandwidth=BW, device="cuda")
        model.fit(
            (train_X_deep.cuda(), train_y), 
            (test_X_deep.cuda(), test_y), 
            loader=False, 
            iters=1,
            classif=True,
            reg=-1,
            M_batch_size=10000
        )
        M = model.M
        
        
    if KERNEL == 'gaussian':
        model = rfm.GaussRFM(bandwidth=BW, device="cuda")
        model.fit(
            (train_X_deep.cuda(), train_y), 
            (test_X_deep.cuda(), test_y), 
            loader=False, 
            iters=1,
            classif=True,
            reg=-1,
            M_batch_size=10000
        )
        M = model.M
    
    test_X_deep = test_X_deep.cuda()
    train_X_deep = train_X_deep.cuda()
    sqrtM = matrix_sqrt(M).cuda()
    MXs.append((train_X_deep @ sqrtM).cpu())
    
    _, d = train_X_deep.shape

    if use_rff:
        dim = train_X_deep.shape[1]
        # encoding = rff.layers.GaussianEncoding(sigma=SIGMA, input_size=dim, encoded_size=WIDTH).cuda().to(train_X_deep.dtype)
        # feature_fn = lambda x: encoding(x@sqrtM)
        rff_model = rff_laplace.RFF(D=dim, gamma=SIGMA) #gamma=0.1 for cifar/svhn, 0.05 for MNIST
        rff_model.fit(train_X_deep)
        feature_fn = lambda x: rff_model.transform(x@sqrtM)
    else:
        feature_fn = sample_feats(sqrtM, d, WIDTH)
        feature_fn = feature_fn.cpu()
        
       
    with torch.no_grad():
        if not use_rff:
            feature_fn = feature_fn.cuda()
            
        train_X_deep = feature_fn(train_X_deep)
        test_X_deep = feature_fn(test_X_deep)
        
        if not use_rff:
            feature_fn = feature_fn.cpu()
            
    M = M.cpu()
    sqrtM = sqrtM.cpu()
    del model, M, sqrtM, feature_fn
    
    PhiMXs.append(train_X_deep.cpu().clone())
    
    torch.cuda.empty_cache()


## NC1 and NC2, SVD Plots ##

# Xs, MXs, PhiMXs
right_covsW_B = []
left_covsW_B = []
base_covsW_B = []

left_covs_mu = []
right_covs_mu = []
base_covs_mu = []

for layer in range(DEPTH):
    
    X = Xs[layer].cuda()
    Xr = MXs[layer].cuda()
    Xl = PhiMXs[layer].cuda()
    
    SigmaB, SigmaW, _, mus  = get_class_variance(X, train_y)
    covB, covW = SigmaB.trace().item(), SigmaW.trace().item()
    
    SigmaBr, SigmaWr, _, mus_r = get_class_variance(Xr, train_y)
    covBr, covWr = SigmaBr.trace().item(), SigmaWr.trace().item()

    SigmaBl, SigmaWl, _, mus_l = get_class_variance(Xl, train_y)
    covBl, covWl = SigmaBl.trace().item(), SigmaWl.trace().item()
    
    base_covsW_B.append(covW/covB)
    right_covsW_B.append(covWr/covBr)
    left_covsW_B.append(covWl/covBl)

    base_covs_mu.append(get_NC2_from_means(mus))
    right_covs_mu.append(get_NC2_from_means(mus_r))
    left_covs_mu.append(get_NC2_from_means(mus_l))
    
    del X, Xr, Xl

run_path = f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_deeprfm'
results_path = os.path.join('results', run_path)

try:
    os.mkdir(results_path)
except:
    pass

right_nc1_path = os.path.join(results_path, f'right_nc1_seed_{SEED}.pt')
left_nc1_path = os.path.join(results_path, f'left_nc1_seed_{SEED}.pt')
base_nc1_path = os.path.join(results_path, f'base_nc1_seed_{SEED}.pt')

torch.save(right_covsW_B, right_nc1_path)
torch.save(left_covsW_B, left_nc1_path)
torch.save(base_covsW_B, base_nc1_path)

right_nc2_path = os.path.join(results_path, f'right_nc2_seed_{SEED}.pt')
left_nc2_path = os.path.join(results_path, f'left_nc2_seed_{SEED}.pt')
base_nc2_path = os.path.join(results_path, f'base_nc2_seed_{SEED}.pt')

torch.save(right_covs_mu, right_nc2_path)
torch.save(left_covs_mu, left_nc2_path)
torch.save(base_covs_mu, base_nc2_path)

fig1, ax1 = plt.subplots(1, 1)
fig1_log, ax1_log = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)
fig2_log, ax2_log = plt.subplots(1, 1)

ax1.grid()
ax2.grid()
ax1_log.grid()
ax2_log.grid()
ax1.set_xlim(1,20)
ax2.set_xlim(1,20)
ax1_log.set_xlim(1,20)
ax2_log.set_xlim(1,20)

xsteps = np.arange(0,DEPTH) + 1
ax1_log.semilogy(xsteps, left_covsW_B, label="left")
ax1_log.semilogy(xsteps, right_covsW_B, label="right")
ax1_log.semilogy(xsteps, base_covsW_B, label="none")

ax1.plot(xsteps, left_covsW_B, label="left")
ax1.plot(xsteps, right_covsW_B, label="right")
ax1.plot(xsteps, base_covsW_B, label="none")

ax2_log.semilogy(xsteps, left_covs_mu, label="left")
ax2_log.semilogy(xsteps, right_covs_mu, label="right")
ax2_log.semilogy(xsteps, base_covs_mu, label="none")

ax2.plot(xsteps, left_covs_mu, label="left")
ax2.plot(xsteps, right_covs_mu, label="right")
ax2.plot(xsteps, base_covs_mu, label="none")

xsteps = np.arange(0,DEPTH) + 1
xticklabels = [str(x) for x in xsteps]
ax1_log.set_xticks(xsteps)
ax1_log.set_xticklabels(xticklabels)
ax2_log.set_xticks(xsteps)
ax2_log.set_xticklabels(xticklabels)
ax1.set_xticks(xsteps)
ax1.set_xticklabels(xticklabels)
ax2.set_xticks(xsteps)
ax2.set_xticklabels(xticklabels)

ax1.set_xlabel("Layer")
ax2.set_xlabel("Layer")
ax1_log.set_xlabel("Layer")
ax2_log.set_xlabel("Layer")

ax1.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_B)$', rotation=0)
ax1.yaxis.set_label_coords(-0.4,0.4)

ax1_log.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_B)$', rotation=0)
ax1_log.yaxis.set_label_coords(-0.4,0.4)

ax2.set_ylabel(r'$\left\|\tilde{\mu}\tilde{\mu}^\top - \Sigma_{\mathrm{ETF}}\right\|$', rotation=0)
ax2.yaxis.set_label_coords(-0.4,0.4)

ax2_log.set_ylabel(r'$\left\|\tilde{\mu}\tilde{\mu}^\top - \Sigma_{\mathrm{ETF}}\right\|$', rotation=0)
ax2_log.yaxis.set_label_coords(-0.4,0.4)


handles1, _ = ax1.get_legend_handles_labels()
handles2, _ = ax2.get_legend_handles_labels()

handles1_log, _ = ax1_log.get_legend_handles_labels()
handles2_log, _ = ax2_log.get_legend_handles_labels()

labels = [r'$\Phi(M^{1/2} X)$',
          r'$M^{1/2} X$',
          r'$X$'
        ]

fig1_log.legend(handles1_log, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncols=1)
fig2_log.legend(handles2_log, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncols=1)

fig1.legend(handles1, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncols=1)
fig2.legend(handles2, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncols=1)

fig1_log.set_size_inches(6, 3)
fig2_log.set_size_inches(6, 3)
fig1.set_size_inches(6, 3)
fig2.set_size_inches(6, 3)


print("artifact directory:",dir_path)
try:
    print("Making directory")
    os.mkdir(dir_path)
except:
    pass

fig1_log.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_seed_{SEED}_deeprfm_nc1_log.pdf'), format='pdf', bbox_inches="tight")
fig2_log.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_seed_{SEED}_deeprfm_nc2_log.pdf'), format='pdf', bbox_inches="tight")

fig1.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_seed_{SEED}_deeprfm_nc1.pdf'), format='pdf', bbox_inches="tight")
fig2.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_seed_{SEED}_deeprfm_nc2.pdf'), format='pdf', bbox_inches="tight")

        

# for idx, X in enumerate(Xs):
#     try:
#         plt.close()
#     except:
#         pass
    
#     if idx%2==1:
#         continue
        
    
        
#     fig, ax = plt.subplots(1, 1)
#     # XX = batch_multiply(X.cuda())
#     with torch.no_grad():
#         X = X - X.mean(dim=0).unsqueeze(0)
#         X /= X.norm(dim=1).unsqueeze(1)
#         XXt = batch_multiply(X)
    
#     # im = ax.imshow(X@X.T)
#     im = ax.imshow(XXt)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f'Layer {idx+1}', fontsize=20)
    
#     if idx==18:
#         fig.colorbar(im, orientation='vertical')
    
#     plt.savefig(os.path.join(dir_path, f'deep_rfm_vis_{dataset}_rff_{use_rff}_sigma_{SIGMA}_n_{n}_layer_{idx}.pdf'), format='pdf')
#     del im, XXt
    
# for idx, X in enumerate(MXs):
#     try:
#         plt.close()
#     except:
#         pass
    
#     if idx%2==1:
#         continue
        
#     print("Making figs")
#     fig, ax = plt.subplots(1, 1)
#     # XX = batch_multiply(X.cuda())
#     with torch.no_grad():
#         print("Centering")
#         X = X - X.mean(dim=0).unsqueeze(0)
#         print("Dividing")
#         X /= X.norm(dim=1).unsqueeze(1)
#         print("Batch multiplying")
#         XXt = batch_multiply(X)
    
#     # im = ax.imshow(X@X.T)
#     print("Showing")
#     im = ax.imshow(XXt)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f'Layer {idx+1}', fontsize=20)
    
#     print("Colorbar")
#     if idx==18:
#         fig.colorbar(im, orientation='vertical')
    
#     print("Saving")
#     plt.savefig(os.path.join(dir_path, f'deep_rfm_vis_{dataset}_rff_{use_rff}_sigma_{SIGMA}_n_{n}_layer_{idx}_MXs.pdf'), format='pdf')
    
    
#     del im, XXt, fig, ax

