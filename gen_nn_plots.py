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


import random


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=50000)
parser.add_argument('-dataset', default="cifar")
parser.add_argument('-epochs', default=300)
parser.add_argument('-depth', default=5)
parser.add_argument('-weight_decay', default=0.0)
parser.add_argument('-momentum', default=0.0)
parser.add_argument('-lr', default=0.05)
parser.add_argument('-init', default=1.0)
parser.add_argument('-opt', default='sgd')
parser.add_argument('-measure_every', default=10)
parser.add_argument('-model', default='mlp')
parser.add_argument('-width', type=int, default=512)
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
MEASURE_EVERY = int(args.measure_every)
model_type = args.model

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

dir_name = f'{model_type}_{dataset}_n_{n}_lr_{LR}_init_{init}_depth_{NUM_LAYERS}_wd_{WD}_mom_{momentum}'
save_path = os.path.join('nn_figures',dir_name)
results_path = os.path.join('nn_results',dir_name)
 
fs=14
fs2=12

#### Generate NFA fig ####

fig_title = os.path.join(save_path, f'nfa_sqrt_agop_all_seeds.pdf')
fig, axes = plt.subplots(1,3)
ax1, ax2, ax3 = axes


normalized_covB = []
normalized_covW = []
losses = []
sqrt_nfas = []

for SEED in [0,1,2]:
    loss_path = os.path.join(results_path, f'losses_seed_{SEED}.pt')
    covW_path = os.path.join(results_path, f'covW_seed_{SEED}.pt')
    covB_path = os.path.join(results_path, f'covB_seed_{SEED}.pt')

    normalized_covB.append(torch.tensor(torch.load(covB_path)))
    normalized_covW.append(torch.tensor(torch.load(covW_path)))
    losses.append(torch.tensor(torch.load(loss_path)))
    
    sqrt_nfa = []
    for layer in range(NUM_LAYERS):
        nfa_path = os.path.join(results_path, f'sqrt_nfa_layer_{layer}_seed_{SEED}.pt')
        sqrt_nfa.append(torch.tensor(torch.load(nfa_path)))
    sqrt_nfa = torch.stack(sqrt_nfa, dim=0)
    sqrt_nfas.append(sqrt_nfa)


normalized_covB = torch.stack(normalized_covB)
normalized_covW = torch.stack(normalized_covW)
losses = torch.stack(losses)
sqrt_nfas = torch.stack(sqrt_nfas) # (num_seeds, num_layers, num_measurements)

covB_mean = normalized_covB.mean(dim=0)
covW_mean = normalized_covW.mean(dim=0)
losses_mean = losses.mean(dim=0)

covB_std = normalized_covB.std(dim=0)
covW_std = normalized_covW.std(dim=0)
losses_std = losses.std(dim=0)

sqrt_nfas_mean = sqrt_nfas.mean(dim=0)
sqrt_nfas_std = sqrt_nfas.std(dim=0)

print("covB_mean", covB_mean.shape, "len nfas",sqrt_nfas_mean.shape,"losses", losses_mean.shape)
ax1.semilogy(np.arange(0, len(covB_mean)*MEASURE_EVERY, MEASURE_EVERY), 
             covB_mean, 'b', label=r'$tr(\Sigma^L_B) / tr(\Sigma^L_T)$')
ax1.semilogy(np.arange(0, len(covW_mean)*MEASURE_EVERY, MEASURE_EVERY), 
             covW_mean, 'g', label=r'$tr(\Sigma^L_W) / tr(\Sigma^L_T)$')
ax3.semilogy(losses_mean, 'b')

ax1.fill_between(np.arange(0, len(covB_mean)*MEASURE_EVERY, MEASURE_EVERY), 
                 covB_mean+covB_std, covB_mean-covB_std, color='b', alpha=0.1)
ax1.fill_between(np.arange(0, len(covW_mean)*MEASURE_EVERY, MEASURE_EVERY), 
                 covW_mean+covW_std, covW_mean-covW_std, color='g', alpha=0.1)
ax3.fill_between(np.arange(len(losses_mean)), losses_mean+losses_std, losses_mean-losses_std, color='b', alpha=0.1)

print("Final NFA means per layer:",torch.mean(sqrt_nfas[:,:,-1]),"std dev:",torch.std(sqrt_nfas[:,:,-1]))

colors = ['b','g','r','m','y']
for layer in range(NUM_LAYERS):
    nfa = sqrt_nfas_mean[layer]
    std = sqrt_nfas_std[layer]
    xsteps = np.arange(0, len(nfa)*MEASURE_EVERY, MEASURE_EVERY)
    print("xsteps",xsteps,"nfa",nfa.shape)
    ax2.plot(xsteps, nfa, label=f'Layer {layer+1}', color=colors[layer])
    ax2.fill_between(xsteps, nfa-std, nfa+std, color=colors[layer], alpha=0.1)

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

ax1.grid()
ax2.grid()
ax3.grid()


# fig.suptitle("Normalized feature variance throughout training", fontsize=fs)
fig.set_size_inches(24, 6)
fig.savefig(fig_title, format='pdf')



####### NC1 Figure #######

try:
    plt.close()
except:
    pass

fig1, axes1 = plt.subplots(1, NUM_LAYERS, sharey=True)
fig1_log, axes1_log = plt.subplots(1, NUM_LAYERS, sharey=True)


        
ymax_nc1 = 0
ymax_nc2 = 0
for layer in range(0, NUM_LAYERS):
    
    left_covsW_B = []
    right_covsW_B = []
    base_covsW_B = []
    for SEED in [0,1,2]:
        left_fname = os.path.join(results_path, f'seed_{SEED}_layer_{layer}_nc1_left.pt')
        right_fname = os.path.join(results_path, f'seed_{SEED}_layer_{layer}_nc1_right.pt')
        base_fname = os.path.join(results_path, f'seed_{SEED}_layer_{layer}_nc1_base.pt')

        left_covsW_B.append(torch.tensor(torch.load(left_fname)))
        right_covsW_B.append(torch.tensor(torch.load(right_fname)))
        base_covsW_B.append(torch.tensor(torch.load(base_fname)))
        
    
    left_covsW_B = torch.stack(left_covsW_B)
    right_covsW_B = torch.stack(right_covsW_B)
    base_covsW_B = torch.stack(base_covsW_B) # (num_seeds, num_layers, num_measurements)
    
    left_mean = left_covsW_B.mean(dim=0)
    right_mean = right_covsW_B.mean(dim=0)
    base_mean = base_covsW_B.mean(dim=0)
    
    left_std = left_covsW_B.std(dim=0)
    right_std = right_covsW_B.std(dim=0)
    base_std = base_covsW_B.std(dim=0)

    ymax_nc1 = max([ymax_nc1, left_mean.max().item(), right_mean.max().item(), base_mean.max().item()])
    xsteps = torch.arange(0, len(left_mean)*MEASURE_EVERY, MEASURE_EVERY).cpu()
    # print("len(left_mean)", len(left_mean), "MEASURE_EVERY", MEASURE_EVERY, "xsteps",xsteps)
    
    ax1_log = axes1_log[layer]
    ax1_log.semilogy(xsteps, left_mean, 'r', label="left")
    ax1_log.semilogy(xsteps, right_mean, 'g', label="right")
    ax1_log.semilogy(xsteps, base_mean, 'b', label="none")
    
    left_lower = torch.max(left_mean - left_std, left_mean/20)
    right_lower = torch.max(right_mean - right_std, right_mean/20)
    base_lower = torch.max(base_mean - base_std, base_mean/20)
    
    ax1_log.fill_between(xsteps, left_lower, left_mean+left_std, color='r', alpha=0.1)
    ax1_log.fill_between(xsteps, right_lower, right_mean+right_std, color='g', alpha=0.1)
    ax1_log.fill_between(xsteps, base_lower, base_mean+base_std, color='b', alpha=0.1)
    
    ax1 = axes1[layer]
    ax1.plot(xsteps, left_mean, label="left")
    ax1.plot(xsteps, right_mean, label="right")
    ax1.plot(xsteps, base_mean, label="none")
    
    left_lower = torch.max(left_mean - left_std, left_mean/10)
    right_lower = torch.max(right_mean - right_std, right_mean/10)
    base_lower = torch.max(base_mean - base_std, base_mean/10)
    
    ax1.fill_between(xsteps, left_lower, left_mean+left_std, color='r', alpha=0.1)
    ax1.fill_between(xsteps, right_lower, right_mean+right_std, color='g', alpha=0.1)
    ax1.fill_between(xsteps, base_lower, base_mean+base_std, color='b', alpha=0.1)
         
    ax1.set_title(f'Layer {layer+1}')
    ax1_log.set_title(f'Layer {layer+1}')
    
    ax1.grid()
    ax1_log.grid()
    

    if layer==0:
        ax1.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_B)$', rotation=0)
        ax1.yaxis.set_label_coords(-0.5,0.4)
        
        ax1_log.set_ylabel(r'$tr(\Sigma_W) / tr(\Sigma_B)$', rotation=0)
        ax1_log.yaxis.set_label_coords(-0.6,0.4)
        
    
    step = (len(left_mean)-1)*MEASURE_EVERY // 5
    xticks = np.arange(0, (len(left_mean)-1)*MEASURE_EVERY + step, step)
    
    ax1.set_xticks(xticks)
    ax1_log.set_xticks(xticks)
    
    ax1.set_xlabel("Epochs")
    ax1_log.set_xlabel("Epochs")
    

if dataset == 'mnist':
    yticks = [1e1,1e0,1e-1,1e-2,1e-3]
elif dataset == 'svhn':
    yticks = [1e2, 1e1,1e0,1e-1,1e-2]
elif model_type=='resnet18' and dataset == 'cifar':
    yticks = [1e2, 1e1,1e0,1e-1,1e-2]
elif dataset == 'cifar':
    yticks = [1e1,1e0,1e-1,1e-2]
    
yticklabels = [str(x) for x in yticks]
axes1_log[0].set_yticks(yticks)
axes1_log[0].set_yticklabels(yticklabels)        

handles1, _ = axes1[-1].get_legend_handles_labels()
handles1_log, _ = axes1_log[-1].get_legend_handles_labels()
labels = [r'$\mathrm{ReLU} (U S V^\top \Phi(X))$',
        r'$S V^\top \Phi(X)$',
        r'$\Phi(X)$']

fig1_log.legend(handles1_log, labels, loc="lower center", bbox_to_anchor=(0.5, -0.175), fontsize=8, ncols=4)
fig1.legend(handles1, labels, loc="lower center", bbox_to_anchor=(0.5, -0.175), fontsize=8, ncols=4)

fig1_log.set_size_inches(3*NUM_LAYERS, 3)
fig1.set_size_inches(3*NUM_LAYERS, 3)

# plt.show()
fig1_log.savefig(os.path.join(save_path, f'svd_nc1_log_all_seeds.pdf'), format='pdf', bbox_inches="tight")
fig1.savefig(os.path.join(save_path, f'svd_nc1_all_seeds.pdf'), format='pdf', bbox_inches="tight")
