import os
from copy import deepcopy
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.func import jacrev, vmap



torch.cuda.manual_seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=1000)
parser.add_argument('-dataset', default="cifar")
parser.add_argument('-epochs', default=1000)
parser.add_argument('-depth', default=3)
parser.add_argument('-weight_decay', default=0.0)
parser.add_argument('-momentum', default=0.0)
parser.add_argument('-lr', default=0.01)
parser.add_argument('-init', default=1.0)
parser.add_argument('-opt', default='sgd')
parser.add_argument('-measure_every', default=25)
parser.add_argument('-model', default='mlp')
args = parser.parse_args()

for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/"

dataset = args.dataset
n = int(args.n)
NUM_LAYERS = int(args.depth)
NUM_EPOCHS = int(args.epochs)
WD = float(args.weight_decay)
momentum = float(args.momentum)
LR = float(args.lr)
init = float(args.init)
OPT = args.opt
MEASURE_EVERY = int(args.measure_every)

print("LR",LR)

model_type = args.model
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

def get_classes(dataset, classes):
    
    sub_dataset = []
    for ix in range(len(dataset)):
        X,y = dataset[ix]
        
        if y in classes:
            newc = classes.index(y)
            sub_dataset.append((X,newc))

    return sub_dataset

def preprocess(X):
    """
    X : (n, c, P, Q)
    """
    Xpp = X - X.mean(dim=0)
    Xpp /= X.std(dim=0).unsqueeze(0)
    return torch.nan_to_num(Xpp)

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


if model_type == 'mlp':
    train_X = train_X_.reshape(len(train_X_),-1)#.double()
    test_X = test_X_.reshape(len(test_X_),-1)#.double()
else:
    train_X = train_X_
    test_X = test_X_
    
train_y = train_y_.to(train_X_.dtype)
test_y = test_y_.to(test_X_.dtype)


train_X = preprocess(train_X)
test_X = preprocess(test_X)

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
    
mb_size = 128
train_loader = torch.utils.data.DataLoader(MyDataset(train_X, train_y), batch_size=mb_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(MyDataset(test_X, test_y), batch_size=mb_size, shuffle=True)


def matrix_sqrt(M):
    S, V = torch.linalg.eigh(M)
    S[S<0] = 0
    S = torch.diag(S**0.5)
    return V @ S @ V.T

def fix_speeds(model, layer_to_fix, C=5, eps=1e-1):
    if layer_to_fix == 0:
        eps = 0.
        C = 200
    
    for i, layer in enumerate(model.layers):
        
        if isinstance(layer, nn.Linear):
            linear_layer = layer
          
        else:
            linear_layer = layer[0]
            
        linear_layer.weight.grad /= (eps + linear_layer.weight.grad.norm())
        if i==layer_to_fix:
            linear_layer.weight.grad *= C
        else:
            linear_layer.weight.grad /= C
        
        # linear_layer.weight.grad /= linear_layer.weight.grad.norm()
        # linear_layer.weight.grad *= linear_layer.weight.norm()
            
            
def train_step(net, lr, WD, momentum, train_loader, layer_to_fix=None, warmup=False, opt='sgd'):
    net.train()
    train_loss = 0.
    num_batches = len(train_loader)
    
    if opt=='sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=WD, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WD)
    
    for batch_idx, batch in enumerate(train_loader):
        
        if warmup:
            lr_ = lr * batch_idx/num_batches 
            if opt=='sgd':
                optimizer = torch.optim.SGD(net.parameters(), lr=lr_, weight_decay=WD, momentum=momentum)
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr_, weight_decay=WD)
        
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs)).float()
        target = Variable(targets).float()
        loss = criterion(output, target)
        loss.backward()
        
        # if layer_to_fix is not None:
        # layer_to_fix = 0
        # fix_speeds(net, layer_to_fix, C=20, eps=2.5e-3)
        
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
        return nn.ReLU()(x)
        
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


def measure_agop(net, X):
    def get_jacobian(X_):
        def fnet_single(x_):
            return net(x_.unsqueeze(0)).squeeze(0)
        return vmap(jacrev(fnet_single))(X_).detach()
    
    
    n, d = X.shape
    grad = get_jacobian(X)
    grad = grad.reshape(-1, d)
    agop = grad.T @ grad

    return agop

def getW(model, layer):
    model_layer = model.layers[layer]
    return model_layer[0].weight.data

def get_fmap(net_, last_layer):
    if last_layer==0:
        return nn.Identity()
    return nn.Sequential(*net_.layers[:last_layer]).eval()
    
def get_submodel(net_, last_layer):
    return nn.Sequential(*net_.layers[last_layer:]).eval()


def get_fmap_resnet(net_, last_layer):
    resnet = net_[0]
    mlp = net_[1]
    return nn.Sequential(*[resnet, get_fmap(mlp, last_layer)]).eval()

def get_fully_linear_fmap(net_, last_layer):
    if last_layer==0:
        return nn.Identity()
    layers = net_.layers[:last_layer]
    # layers = [layer[0] for layer in layers]
    layers[-1] = layers[-1][0]
    return nn.Sequential(*layers).eval()


PRINT_EVERY = MEASURE_EVERY


def train_network(net, train_loader, test_loader, lr=LR, num_epochs=NUM_EPOCHS):

    params = 0
    for i, param in enumerate(list(net.parameters())):
        size = 1
        for j in range(len(param.size())):
            size *= param.size()[j]
            params += size

    print("NUMBER OF PARAMS: ", params)

    

    losses = []
    
    cov_Bs = []
    cov_Ws = []
    cov_Ts = []
    nfas = []
    sqrt_nfas = []
    all_agops = []
    models = []
    
    best_loss = float("inf")
    prev_train_loss = None
    layer_to_fix = None
    for i in range(num_epochs+1):
        
#         if i==num_epochs//2:
#             lr /= 2
        
#         if i==0:
#             train_loss = train_step(net, lr, WD, momentum, train_loader, layer_to_fix, warmup=True)
#         else:
#             train_loss = train_step(net, lr, WD, momentum, train_loader, layer_to_fix, warmup=False)

        train_loss = train_step(net, lr, WD, momentum, train_loader, layer_to_fix, warmup=False, opt=OPT)
        test_loss = val_step(net, test_loader)
        
        if prev_train_loss is None:
            prev_train_loss = train_loss
            
        losses.append(train_loss)

       
        if test_loss < best_loss:
            best_loss = test_loss
            
        
        
        if i%PRINT_EVERY==0:
            print("Epoch: ", i,
                  "Train Loss: ", train_loss, "Test Loss: ", test_loss,
                  "Best Test Loss: ", best_loss, "Layer to fix: ", layer_to_fix)
            
        if i%MEASURE_EVERY==0:
            cov_B = torch.zeros(NUM_LAYERS).to(train_X.device)
            cov_W = torch.zeros(NUM_LAYERS).to(train_X.device)
            cov_T = torch.zeros(NUM_LAYERS).to(train_X.device)
            nfa = torch.zeros(NUM_LAYERS).to(train_X.device)
            sqrt_nfa = torch.zeros(NUM_LAYERS).to(train_X.device)
            agops_t = []
            for layer in range(NUM_LAYERS):
                with torch.no_grad():
                    if model_type=='mlp':

                        fmap = get_fmap(net, layer)
                        # fmap = get_fully_linear_fmap(net,layer)
                        sub_model = get_submodel(net, layer)

                        phiX = fmap(train_X)

                        with torch.no_grad():
                            agop = measure_agop(sub_model, phiX)
                        
                        sqrt_agop = matrix_sqrt(agop)

                        W = getW(net, layer)
                        nfm = W.T@W
                        
                        agops_t.append(agop.cpu())

                        nfa[layer] = mat_cov(agop, nfm)
                        sqrt_nfa[layer] = mat_cov(sqrt_agop, nfm)

                        fmap_next = get_fmap(net, layer+1)
                        # fmap_next = get_fully_linear_fmap(net,layer+1)
                        phiX_next = fmap_next(train_X)
                        
                        SigmaB, SigmaW, SigmaT, _ = get_class_variance(phiX_next, train_y)
                        cov_B[layer], cov_W[layer], cov_T[layer] = SigmaB.trace(), SigmaW.trace(), SigmaT.trace()

                    elif model_type=='resnet18':

                        fmap = get_fmap_resnet(net, layer)
                        with torch.no_grad():
                            phiX = fmap(train_X)

                        sub_model = get_submodel(net[1], layer)
                        
                        with torch.no_grad():
                            agop = measure_agop(sub_model, phiX)
                        sqrt_agop = matrix_sqrt(agop)

                        W = getW(net[1], layer)
                        nfm = W.T@W
                        
                        agops_t.append(agop.cpu())

                        nfa[layer] = mat_cov(agop, nfm)
                        sqrt_nfa[layer] = mat_cov(sqrt_agop, nfm)

                        fmap_next = get_fmap_resnet(net, layer+1)
                        phiX_next = fmap_next(train_X)
                        SigmaB, SigmaW, SigmaT, _ = get_class_variance(phiX_next, train_y)
                        cov_B[layer], cov_W[layer], cov_T[layer] = SigmaB.trace(), SigmaW.trace(), SigmaT.trace()


            cov_Bs.append(cov_B)
            cov_Ws.append(cov_W)
            cov_Ts.append(cov_T)
            nfas.append(nfa)
            sqrt_nfas.append(sqrt_nfa)
            all_agops.append(agops_t)
            
            net.cpu()
            models.append(deepcopy(net).eval())
            net.cuda()
            
            # print("nfa", nfa)
            layer_to_fix = torch.argmin(nfa).item()
            # print("layer to fix", layer_to_fix)
            
    return net, losses, cov_Bs, cov_Ws, cov_Ts, models, nfas, sqrt_nfas, all_agops



if model_type=='mlp':
    inits = [init for _ in range(NUM_LAYERS)]
    d = train_X.shape[1]
    k = 512
    layers = [
                nn.Sequential(nn.Linear(d, k, bias=False), 
                                # nn.BatchNorm1d(k),
                              Activation()
                             )
             ]
    nn.init.xavier_uniform_(layers[-1][0].weight, gain=inits[0])

    for i in range(NUM_LAYERS - 1):
        layers += [
                    nn.Sequential(nn.Linear(k, k, bias=False), 
                                    # nn.BatchNorm1d(k),
                                  Activation()
                                 )
                ]
        nn.init.xavier_uniform_(layers[-1][0].weight, gain=inits[1+i])

    layers += [nn.Linear(k, NUM_CLASSES, bias=False)]

    model = MLP(layers)
    model.cuda()
    
elif model_type=='resnet18':
    layers = []
    inits = [init for _ in range(NUM_LAYERS)]
    inits[0] = 0.05
    inits[1] = 0.05
    for i in range(NUM_LAYERS):
        layers += [
                    nn.Sequential(nn.Linear(512, 512, bias=False),
                                  Activation()
                                 )
                ]
        nn.init.xavier_uniform_(layers[-1][0].weight, gain=inits[i])
    layers += [nn.Linear(512, NUM_CLASSES, bias=False)]
    mlp = MLP(layers)
        
    resnet = torchvision.models.resnet18(weights=None)
    
    _, c, _, _ = train_X.shape
    num_channels = resnet.conv1.weight.shape[0]
    resnet.conv1 = nn.Conv2d(c, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Identity()
    
    model = nn.Sequential(*[resnet, mlp])
    model.cuda()

print(model)

model, losses, cov_Bs, cov_Ws, cov_Ts, models, nfas, sqrt_nfas, all_agops = train_network(model, train_loader, test_loader)



import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

dir_name = f'{model_type}_{dataset}_n_{n}_lr_{LR}_init_{init}_depth_{NUM_LAYERS}_wd_{WD}_mom_{momentum}'
save_path = os.path.join('nn_figures',dir_name)

if not os.path.isdir(save_path):
    os.mkdir(save_path)

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

# scale = max( np.max(normalized_covB), np.max(normalized_covW))
normalized_covW /= normalized_covT
normalized_covB /= normalized_covT

for use_sqrt in [True, False]:
    if use_sqrt:
        fig_title = os.path.join(save_path, 'nfa_sqrt_agop.pdf')
    else:
        fig_title = os.path.join(save_path, 'nfa_full_agop.pdf')
        
    fig, axes = plt.subplots(1,3)
    ax1, ax2, ax3 = axes
    ax1.semilogy(np.arange(0, len(normalized_covB)*MEASURE_EVERY, MEASURE_EVERY), normalized_covB, label=r'$tr(\Sigma^L_B) / tr(\Sigma^L_T)$')
    ax1.semilogy(np.arange(0, len(normalized_covW)*MEASURE_EVERY, MEASURE_EVERY), normalized_covW, label=r'$tr(\Sigma^L_W) / tr(\Sigma^L_T)$')
    ax3.semilogy(losses)

    for layer in range(NUM_LAYERS):
        if use_sqrt:
            nfa = [x[layer].cpu() for x in sqrt_nfas]
            ax2.plot(np.arange(0, len(nfa)*MEASURE_EVERY, MEASURE_EVERY), nfa, label=f'Layer {layer+1}')
        else:
            nfa = [x[layer].cpu() for x in nfas]
            ax2.plot(np.arange(0, len(nfa)*MEASURE_EVERY, MEASURE_EVERY), nfa, label=f'Layer {layer+1}')

    ax2.set_yticks(torch.linspace(0,1,11))

    ax1.set_ylabel('Value', fontsize=fs)
    ax2.set_ylabel('Correlation', fontsize=fs)
    ax3.set_ylabel('Loss', fontsize=fs)

    yticks = [1e1,1e0,1e-1,1e-2]
    yticklabels = [str(x) for x in yticks]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)


    yticks = [1e0,1e-1,1e-2]
    yticklabels = [str(x) for x in yticks]
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)

    step = NUM_EPOCHS // 5
    xticks = np.arange(0, NUM_EPOCHS + step, step)
    for ax in axes[:2]:
        ax.set_xticks(xticks)

    ax1.legend(fontsize=fs2, ncols=1)
    print("Lower right legend")
    ax2.legend(fontsize=fs2, ncols=3, loc='lower right')

    for ax in axes:
        ax.set_xlabel("Epochs", fontsize=fs)

    # fig.suptitle("Normalized feature variance throughout training", fontsize=fs)
    fig.set_size_inches(24, 6)
    fig.savefig(fig_title, format='pdf')


# ## NC1 and NC2, SVD Plots


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

# def get_NC2_yyT_from_means(u):
#     u = u / u.norm(dim=1).unsqueeze(1)
#     uuT = u@u.T
    
    
#     mean_off_diag = (uuT.sum() - uuT.diag().sum()).item() / NUM_CLASSES / (NUM_CLASSES-1)
#     off_diag = torch.ones_like(uuT)*mean_off_diag
#     for i in range(NUM_CLASSES):
#         off_diag[i,i] = 0
    
#     uuT -= off_diag
#     return mat_cov(uuT, torch.eye(NUM_CLASSES).to(u.device))


# ## svd measurements with agop

# In[47]:


try:
    plt.close()
except:
    pass

fig1, axes1 = plt.subplots(1, NUM_LAYERS, sharey=True)
fig1_log, axes1_log = plt.subplots(1, NUM_LAYERS, sharey=True)
fig2, axes2 = plt.subplots(1, NUM_LAYERS, sharey=True)
fig2_log, axes2_log = plt.subplots(1, NUM_LAYERS, sharey=True)

xsteps = torch.arange(0, len(models)*MEASURE_EVERY, MEASURE_EVERY).cpu()

ymax_nc1 = 0
ymax_nc2 = 0
for layer in range(0, NUM_LAYERS):
    
    right_covsW_B = []
    left_covsW_B = []
    base_covsW_B = []
    agop_covsW_B = []
    
    left_covs_mu = []
    right_covs_mu = []
    base_covs_mu = []
    agop_covs_mu = []
    
    for model_idx, net in tqdm(enumerate(models)):
        # print("Model", model_idx,"layer", layer)
        net.cuda()
        if model_type=='mlp':
            W = getW(net, layer)
            # fmap = get_fmap(net, layer)
            fmap = get_fully_linear_fmap(net, layer)
        elif model_type=='resnet18':
            W = getW(net[1], layer)
            fmap = get_fmap_resnet(net, layer)
            net = net[1]
        
        with torch.no_grad():
            phiX = fmap(train_X)
        
        # right calcs
        SigmaB, SigmaW, _, mus  = get_class_variance(phiX, train_y)
        covB, covW = SigmaB.trace().item(), SigmaW.trace().item()
        
        _, S, Vt = torch.linalg.svd(W)
        S = torch.diag(S)
        if S.shape[1] != Vt.shape[0]:
            newS = torch.zeros(S.shape[0], Vt.shape[0]).to(S.device)
            newS[:,:S.shape[1]] = S.clone()
            S = newS
        
        phiXr = phiX@Vt.T@S.T
        SigmaBr, SigmaWr, _, mus_r = get_class_variance(phiXr, train_y)
        covBr, covWr = SigmaBr.trace().item(), SigmaWr.trace().item()
        
        
        # left calcs
        phiXl = net.layers[layer](phiX)
        SigmaBl, SigmaWl, _, mus_l = get_class_variance(phiXl, train_y)
        covBl, covWl = SigmaBl.trace().item(), SigmaWl.trace().item()
        
        
        
        ## agop calcs
        # print(len(all_agops[model_idx]))
        agop = all_agops[model_idx][layer].cuda()
        Sa2, Va = torch.linalg.eigh(agop)
        Sa2[Sa2 < 0] = 0
        Sa = Sa2 ** 0.5
        Sa = torch.diag(Sa)
        
        phiXa = phiX@Va@Sa
        SigmaBa, SigmaWa, _, mus_a = get_class_variance(phiXa, train_y)
        covBa, covWa = SigmaBa.trace().item(), SigmaWa.trace().item()
        
        agop.cpu()
        
        
        
        base_covsW_B.append(covW/covB)
        right_covsW_B.append(covWr/covBr)
        left_covsW_B.append(covWl/covBl)
        agop_covsW_B.append(covWa/covBa)
        
        base_covs_mu.append(get_NC2_from_means(mus))
        right_covs_mu.append(get_NC2_from_means(mus_r))
        left_covs_mu.append(get_NC2_from_means(mus_l))
        agop_covs_mu.append(get_NC2_from_means(mus_a))
        
        net.cpu()
    
    

    ymax_nc1 = max([ymax_nc1] + left_covsW_B + right_covsW_B + base_covsW_B + agop_covsW_B)
    ymax_nc2 = max([ymax_nc2] + left_covs_mu + right_covs_mu + base_covs_mu + agop_covs_mu)
    
    ax1_log = axes1_log[layer]
    ax1_log.semilogy(xsteps, left_covsW_B, label="left")
    ax1_log.semilogy(xsteps, right_covsW_B, label="right")
    ax1_log.semilogy(xsteps, base_covsW_B, label="none")
    # ax1.semilogy(xsteps, agop_covsW_B, label="agop")
    
    ax1 = axes1[layer]
    ax1.plot(xsteps, left_covsW_B, label="left")
    ax1.plot(xsteps, right_covsW_B, label="right")
    ax1.plot(xsteps, base_covsW_B, label="none")
    
    ax2_log = axes2_log[layer]
    ax2_log.semilogy(xsteps, left_covs_mu, label="left")
    ax2_log.semilogy(xsteps, right_covs_mu, label="right")
    ax2_log.semilogy(xsteps, base_covs_mu, label="none")
    # ax2.semilogy(xsteps, agop_covs_mu, label="agop")
    
    ax2 = axes2[layer]
    ax2.plot(xsteps, left_covs_mu, label="left")
    ax2.plot(xsteps, right_covs_mu, label="right")
    ax2.plot(xsteps, base_covs_mu, label="none")
    
    
    ax1.set_title(f'Layer {layer+1}')
    ax2.set_title(f'Layer {layer+1}')
    ax1_log.set_title(f'Layer {layer+1}')
    ax2_log.set_title(f'Layer {layer+1}')
    
    # ax.set_yticks(yticks)
    
    if layer==0:
        ax1.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_B)$', rotation=0)
        ax1.yaxis.set_label_coords(-0.5,0.4)
        
        ax1_log.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_B)$', rotation=0)
        ax1_log.yaxis.set_label_coords(-0.6,0.4)
        
        # ax2.set_ylabel(r'$\left\|\tilde{\mu}\tilde{\mu}^\top - (I_K - \frac{1}{K-1}11^\top \right)\right\|$', rotation=0)
        ax2.set_ylabel('NC2', rotation=0)
        ax2.yaxis.set_label_coords(-0.4,0.4)
        
        ax2_log.set_ylabel('NC2', rotation=0)
        ax2_log.yaxis.set_label_coords(-0.4,0.4)
        
    
    step = (len(models)-1)*MEASURE_EVERY // 5
    xticks = np.arange(0, (len(models)-1)*MEASURE_EVERY + step, step)
    
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax1_log.set_xticks(xticks)
    ax2_log.set_xticks(xticks)
    
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax1_log.set_xlabel("Epochs")
    ax2_log.set_xlabel("Epochs")
    

if dataset == 'mnist':
    yticks = [1e1,1e0,1e-1,1e-2,1e-3]
elif dataset == 'svhn':
    yticks = [1e2, 1e1,1e0,1e-1,1e-2]
elif dataset == 'cifar':
    yticks = [1e1,1e0,1e-1,1e-2]
    
yticklabels = [str(x) for x in yticks]
axes1_log[0].set_yticks(yticks)
axes1_log[0].set_yticklabels(yticklabels)

if dataset == 'mnist':
    yticks = [1e0,1e-1,1e-2,1e-3,1e-4]
elif dataset == 'svhn':
    yticks = [1e0,1e-1,1e-2]
elif dataset == 'cifar':
    yticks = [1e0,1e-1,1e-2]
yticklabels = [str(x) for x in yticks]
axes2_log[0].set_yticks(yticks)
axes2_log[0].set_yticklabels(yticklabels)
        

handles1, _ = axes1[-1].get_legend_handles_labels()
handles2, _ = axes2[-1].get_legend_handles_labels()

handles1_log, _ = axes1_log[-1].get_legend_handles_labels()
handles2_log, _ = axes2_log[-1].get_legend_handles_labels()
labels = [r'$\mathrm{ReLU} (U S V^\top \Phi(X))$',
        r'$S V^\top \Phi(X)$',
        r'$\Phi(X)$']
        # r'$S_{\mathrm{AGOP}} V_{\mathrm{AGOP}}^\top \Phi(X)$']

fig1_log.legend(handles1_log, labels, loc="lower center", bbox_to_anchor=(0.5, -0.175), fontsize=8, ncols=4)
fig2_log.legend(handles2_log, labels, loc="lower center", bbox_to_anchor=(0.5, -0.175), fontsize=8, ncols=4)

fig1.legend(handles1, labels, loc="lower center", bbox_to_anchor=(0.5, -0.175), fontsize=8, ncols=4)
fig2.legend(handles2, labels, loc="lower center", bbox_to_anchor=(0.5, -0.175), fontsize=8, ncols=4)

fig1_log.set_size_inches(3*NUM_LAYERS, 3)
fig2_log.set_size_inches(3*NUM_LAYERS, 3)
fig1.set_size_inches(3*NUM_LAYERS, 3)
fig2.set_size_inches(3*NUM_LAYERS, 3)

# plt.show()
fig1_log.savefig(os.path.join(save_path, 'svd_nc1_log.pdf'), format='pdf', bbox_inches="tight")
fig2_log.savefig(os.path.join(save_path, 'svd_nc2_log.pdf'), format='pdf', bbox_inches="tight")

fig1.savefig(os.path.join(save_path, 'svd_nc1.pdf'), format='pdf', bbox_inches="tight")
fig2.savefig(os.path.join(save_path, 'svd_nc2.pdf'), format='pdf', bbox_inches="tight")
