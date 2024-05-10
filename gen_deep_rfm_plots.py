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
args = parser.parse_args()

for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/"

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

run_path = f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_deeprfm'
results_path = os.path.join('results', run_path)

right_covsW_B = []
left_covsW_B = []
base_covsW_B = []

right_covs_mu = []
left_covs_mu = []
base_covs_mu = []

for SEED in [0,1,2]:
    right_nc1_path = os.path.join(results_path, f'right_nc1_seed_{SEED}.pt')
    left_nc1_path = os.path.join(results_path, f'left_nc1_seed_{SEED}.pt')
    base_nc1_path = os.path.join(results_path, f'base_nc1_seed_{SEED}.pt')

    right_covsW_B.append(torch.tensor(torch.load(right_nc1_path)))
    left_covsW_B.append(torch.tensor(torch.load(left_nc1_path)))
    base_covsW_B.append(torch.tensor(torch.load(base_covsW_B)))

    right_nc2_path = os.path.join(results_path, f'right_nc2_seed_{SEED}.pt')
    left_nc2_path = os.path.join(results_path, f'left_nc2_seed_{SEED}.pt')
    base_nc2_path = os.path.join(results_path, f'base_nc2_seed_{SEED}.pt')

    right_covs_mu.append(torch.tensor(torch.load(right_nc2_path)))
    left_covs_mu.append(torch.tensor(torch.load(left_nc2_path)))
    base_covs_mu.append(torch.tensor(torch.load(base_nc2_path)))


right_nc1 = torch.staack(right_covsW_B)
left_nc1 = torch.stack(left_covsW_B)
base_nc1 = torch.stack(base_covs_mu)

right_nc2 = torch.stack(right_covs_mu)
left_nc2 = torch.stack(left_covs_mu)
base_nc2 = torch.stack(base_covs_mu)

right_nc1_mean = right_nc1.mean(dim=0)
left_nc1_mean = left_nc1.mean(dim=0)
base_nc1_mean = base_nc1.mean(dim=0)

right_nc2_mean = right_nc2.mean(dim=0)
left_nc2_mean = left_nc2.mean(dim=0)
base_nc2_mean = base_nc2.mean(dim=0)

right_nc1_std = right_nc1.std(dim=0)
left_nc1_std = left_nc1.std(dim=0)
base_nc1_std = base_nc1.std(dim=0)

right_nc2_std = right_nc2.std(dim=0)
left_nc2_std = left_nc2.std(dim=0)
base_nc2_std = base_nc2.std(dim=0)


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

colors = ['r','g','b']
xsteps = np.arange(0,DEPTH) + 1
ax1_log.semilogy(xsteps, left_nc2_mean, label="left", colors[0])
ax1_log.semilogy(xsteps, right_nc2_mean, label="right", colors[1])
ax1_log.semilogy(xsteps, base_nc2_mean, label="none", colors[2])

ax1_log.fill_between(xsteps, left_nc2_mean - left_nc2_std, left_nc2_mean + left_nc2_std, colors[
ax1_log
ax1_log

ax1.plot(xsteps, left_nc2_mean, label="left", colors[0])
ax1.plot(xsteps, right_nc2_mean, label="right", colors[1])
ax1.plot(xsteps, base_nc2_mean, label="none", colors[2])

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

fig1_log.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_all_seeds_deeprfm_nc1_log.pdf'), format='pdf', bbox_inches="tight")
fig2_log.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_all_seeds_deeprfm_nc2_log.pdf'), format='pdf', bbox_inches="tight")

fig1.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_all_seeds_deeprfm_nc1.pdf'), format='pdf', bbox_inches="tight")
fig2.savefig(os.path.join(dir_path, f'{dataset}_n_{n}_kernel_{KERNEL}_depth_{DEPTH}_rff_{use_rff}_sigma_{SIGMA}_all_seeds_deeprfm_nc2.pdf'), format='pdf', bbox_inches="tight")

        