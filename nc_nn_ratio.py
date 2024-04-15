import os
import argparse
from copy import deepcopy
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.func import jacrev, vmap


torch.manual_seed(0)
torch.cuda.manual_seed(0)


## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/"
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=5000)
parser.add_argument('-dataset', default="cifar")
parser.add_argument('-depth', default=6)
parser.add_argument('-epochs', default=2000)
args = parser.parse_args()

dataset = args.dataset
n = int(args.n)
NUM_LAYERS = int(args.depth)
NUM_EPOCHS = int(args.epochs)
criterion = nn.MSELoss()


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
    for ix in range(len(dataset)):
        X,y = dataset[ix]
        
        if y==c1:
            binary_dataset.append((X,0))
        elif y==c2:
            binary_dataset.append((X,1))

    return binary_dataset

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

NUM_CLASSES = 10

train_X_, train_y_ = one_hot_data(trainset, NUM_CLASSES, num_samples=n)
test_X_, test_y_ = one_hot_data(testset, NUM_CLASSES, num_samples=n)


train_X_ = train_X_.cuda()
train_y_ = train_y_.cuda()
test_X_ = test_X_.cuda()
test_y_ = test_y_.cuda()


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


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
            return len(self.y)

    def __getitem__(self, idx):
            X_mb = self.X[idx]
            y_mb = self.y[idx]
            return (X_mb, y_mb)
    
mb_size = 512
train_loader = torch.utils.data.DataLoader(MyDataset(train_X, train_y), batch_size=mb_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(MyDataset(test_X, test_y), batch_size=mb_size, shuffle=True)



def matrix_sqrt(M, thresh=False):
    S, V = torch.linalg.eigh(M)
    S[S<0] = 0

    if thresh:
        # k = int(3*len(S)//4)
        k = -5
        S[:k] = 0

    S = torch.diag(S**0.5)
    return V @ S @ V.T

def train_step(net, optimizer, train_loader):
    net.train()
    train_loss = 0.
    num_batches = len(train_loader)
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs)).float()
        target = Variable(targets).float()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs))
            target = Variable(targets)
        loss = criterion(output, target)
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
    def forward(self, x):
        # return torch.pow(x, 2)
        return nn.ReLU()(x)
        # return nn.GELU()(x)
        
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, perturb=None):
        if perturb is not None:
            x = x + perturb   
        for layer in self.layers:
            x = layer(x)
        return x


def get_fmap(net_, last_layer):
    if last_layer==0:
        return nn.Identity()
    return nn.Sequential(*net_.layers[:last_layer])
    
def get_submodel(net_, last_layer):
    return nn.Sequential(*net_.layers[last_layer:]) 

def getW(model, layer):
    for i, W in enumerate(model.parameters()):
        if i==layer:
            return W.data.detach()

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
            
    cov_B = torch.trace(SigmaB)
    cov_W = torch.trace(SigmaW)
    cov_T = torch.trace(SigmaT)

    return cov_B, cov_W, cov_T, SigmaB, SigmaW



def measure_agop(net, X):
    def get_jacobian(X_):
        def fnet_single(x_):
            return net(x_.unsqueeze(0)).squeeze(0)
        return vmap(jacrev(fnet_single))(X_).detach()
        
    mb_size = int(1e4)
    agop = 0.
    for Xmb in torch.split(X, mb_size):
        m, d = Xmb.shape
        grad = get_jacobian(Xmb)
        grad = grad.reshape(-1, d)
        agop += grad.T @ grad

    return agop / len(X)



PRINT_EVERY = 25
MEASURE_EVERY = 25

def train_network(net, train_loader, test_loader, num_epochs=NUM_EPOCHS, lr=1e-5):

    params = 0
    for i, param in enumerate(list(net.parameters())):
        size = 1
        for j in range(len(param.size())):
            size *= param.size()[j]
            params += size

    print("NUMBER OF PARAMS: ", params)

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    losses = []
    
    cov_Bs = []
    cov_Ws = []
    cov_Ts = []
    nfas = []
    
    models = []

    best_loss = float("inf")
    for i in range(num_epochs+1):
        #print("Epoch: ", i)
        train_loss = train_step(net, optimizer, train_loader)
        test_loss = val_step(net, test_loader)
        
        losses.append(train_loss)

        if train_loss < 1e-15:
            break
        if test_loss < best_loss:
            best_loss = test_loss
        
        if i%PRINT_EVERY==0:
            print("Epoch: ", i,
                  "Train Loss: ", train_loss, "Test Loss: ", test_loss,
                  "Best Test Loss: ", best_loss)
            
        if i%MEASURE_EVERY==0:
            cov_B = torch.zeros(NUM_LAYERS).to(train_X.device)
            cov_W = torch.zeros(NUM_LAYERS).to(train_X.device)
            cov_T = torch.zeros(NUM_LAYERS).to(train_X.device)
            nfa = torch.zeros(NUM_LAYERS).to(train_X.device)
            for layer in range(NUM_LAYERS):
                fmap = get_fmap(net, layer)
                sub_model = get_submodel(net, layer)
                 
                phiX = fmap(train_X)
                
                with torch.no_grad():
                    agop = measure_agop(sub_model, phiX)
                
                W = getW(net, layer)
                nfm = W.T@W
                
                nfa[layer] = mat_cov(agop, nfm)
                cov_B[layer], cov_W[layer], cov_T[layer], _, _ = get_class_variance(phiX, train_y)
                
                del phiX
                
                
            cov_Bs.append(cov_B.detach().cpu())
            cov_Ws.append(cov_W.detach().cpu())
            cov_Ts.append(cov_T.detach().cpu())
            nfas.append(nfa.detach().cpu())
            
            models.append(deepcopy(net).cpu())
            
    return net, losses, cov_Bs, cov_Ws, cov_Ts, models, nfas



d = train_X.shape[1]
k = 512
layers = [
            nn.Sequential(nn.Linear(d, k, bias=False), 
            Activation())
         ]
nn.init.xavier_normal_(layers[-1][0].weight, gain=0.1)

for _ in range(NUM_LAYERS - 1):
    layers += [
                nn.Sequential(nn.Linear(k, k, bias=False), 
                Activation())
            ]

    nn.init.xavier_normal_(layers[-1][0].weight, gain=0.1)

layers += [nn.Linear(k, NUM_CLASSES, bias=False)]

model = MLP(layers)
model.cuda()
model, losses, cov_Bs, cov_Ws, cov_Ts, models, nfas = train_network(model, train_loader, test_loader)



import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
import numpy as np



fs=14
fs2=12

try:
    plt.close()
except:
    pass

layer = 1
normalized_covB = np.array([x[-layer].detach().cpu() for x in cov_Bs])
normalized_covW = np.array([x[-layer].detach().cpu() for x in cov_Ws])
normalized_covT = np.array([x[-layer].detach().cpu() for x in cov_Ts])

normalized_covW /= normalized_covT
normalized_covB /= normalized_covT

fig, axes = plt.subplots(1,3)

ax1, ax2, ax3 = axes

ax1.semilogy(np.arange(0, len(normalized_covB)*MEASURE_EVERY, MEASURE_EVERY), normalized_covB, label=r'$tr(\Sigma^L_B) / tr(\Sigma^L_T)$')
ax1.semilogy(np.arange(0, len(normalized_covW)*MEASURE_EVERY, MEASURE_EVERY), normalized_covW, label=r'$tr(\Sigma^L_W) / tr(\Sigma^L_T)$')

ax3.semilogy(losses)

for layer in range(0, NUM_LAYERS):
    nfa = [x[layer].cpu() for x in nfas]
    ax2.plot(np.arange(0, len(nfa)*MEASURE_EVERY, MEASURE_EVERY), nfa, label=f'Layer {layer}')

    
ax2.set_yticks(torch.linspace(0,1,11))

ax1.set_ylabel('Value', fontsize=fs)
ax2.set_ylabel('Correlation', fontsize=fs)
ax3.set_ylabel('Loss', fontsize=fs)

step = (len(models)-1)*MEASURE_EVERY // 4
xticks = np.arange(0, (len(models)-1)*MEASURE_EVERY + step, step)
for ax in axes[:2]:
    ax.legend(fontsize=fs2)
    ax.set_xticks(xticks)
    
for ax in axes:
    ax.set_xlabel("Epochs", fontsize=fs)
    
fig.suptitle("Normalized feature variance throughout training", fontsize=fs)
fig.set_size_inches(18, 6)
plt.savefig(f'nn_figures/{dataset}_n_{n}_nfa_nc_metrics.pdf', format='pdf')


fig, axes = plt.subplots(1, NUM_LAYERS, sharey=True)

xsteps = torch.arange(0, len(models)*MEASURE_EVERY, MEASURE_EVERY).cpu()

for layer in range(0,layers_to_plot):
    ax1 = axes[0][layer]
    ax2 = axes[1][layer]
    ax3 = axes[2][layer]

    right_covsW = []
    left_covsW = []
    base_covsW = []

    right_covsB = []
    left_covsB = []
    base_covsB = []

    right_covsWB = []
    left_covsWB = []
    base_covsWB = []

    for net in tqdm(models):
        net.cuda()
        W = getW(net, NUM_LAYERS-1-layer)
        fmap = get_fmap(net, NUM_LAYERS-1-layer)
        phiX = fmap(train_X)

        covB, covW, covT, SigmaB, SigmaW = get_class_variance(phiX, train_y)

        U, S, Vt = torch.linalg.svd(W)
        S = torch.diag(S)

        phiXr = phiX@Vt.T@S
        covBr, covWr, covTr, SigmaBr, SigmaWr = get_class_variance(phiXr, train_y)

        phiXl = Activation()(phiX@W.T)
        covBl, covWl, covTl, SigmaBl, SigmaWl = get_class_variance(phiXl, train_y)


        covB = covB.cpu().detach()
        covW = covW.cpu().detach()

        covBr = covBr.cpu().detach()
        covWr = covWr.cpu().detach()
        covBl = covBl.cpu().detach()
        covWl = covWl.cpu().detach()


        base_covsW_B.append(covW/covB)
        right_covsW_B.append(covWr/covBr)
        left_covsW_B.append(covWl/covBl)

        net.cpu()


    ax.semilogy(xsteps, left_covsW, label="left")
    ax.semilogy(xsteps, right_covsW, label="right")
    ax.semilogy(xsteps, base_covsW, label="none")
    ax.set_title(f'Layer {NUM_LAYERS-layer+1}')

    if layer==0:
        ax1.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_T)$', rotation=0)
        ax1.yaxis.set_label_coords(-0.4,0.4)

    step = (len(models)-1)*MEASURE_EVERY // 4
    xticks = np.arange(0, (len(models)-1)*MEASURE_EVERY + step, step)

    ax.set_xticks(xticks)
    ax.set_xlabel("Epochs")

handles, _ = axes[-1].get_legend_handles_labels()
labels = [r'$\mathrm{ReLU} (U S V^\top \Phi(X))$',
        r'$S V^\top \Phi(X)$',
        r'$\Phi(X)$']
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=8, ncols=3)

# fig.suptitle("Neural collapse across layers", y=1.1)
fig.set_size_inches(3*len(datasets), 3*NUM_LAYERS)
plt.savefig(f'nn_figures/{dataset}_n_{n}_svd_nc_metrics.pdf', format='pdf', bbox_inches="tight")




