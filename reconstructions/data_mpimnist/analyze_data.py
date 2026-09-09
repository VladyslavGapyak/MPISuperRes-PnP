import numpy as np

import matplotlib.pyplot as plt

import os
import site
relPathToMPI=os.path.join( os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir, 'utils' )
site.addsitedir(relPathToMPI)

cdir = os.path.dirname(__file__)
path2data = os.path.join(cdir,'recos.npz')

with open(path2data,'rb') as f:
    data = np.load(f)
    u1_all = data['u1']
    u2_all = data['u2']
    losses = data['loss']
    sigmas = data['sigmas']
    mus = data['mu_it']
    S = data['sv']

# compute the PSNR
it_nums, Nx, Ny = u2_all.shape

for i in range(it_nums):

    fig, ax = plt.subplots(2,1)
    ax[0].imshow(u1_all[i,:,:], cmap='gray')
    ax[1].imshow(u2_all[i,:,:], cmap='gray')
    #fig.colorbar(ax=ax)
    plt.savefig(os.path.join(cdir,f'it_{i}.png'),dpi=300,bbox_inches='tight')
    plt.cla()




plt.figure(figsize=(5,5))
plt.semilogy(losses,'b-*')
plt.savefig(os.path.join(cdir,'losses.png'),dpi=300,bbox_inches='tight')
plt.cla()

plt.figure(figsize=(5,5))
plt.semilogy(sigmas,'b-*')
plt.savefig(os.path.join(cdir,'sigmas.png'),dpi=300,bbox_inches='tight')
plt.cla()

plt.figure(figsize=(5,5))
plt.semilogy(mus,'b-*')
plt.savefig(os.path.join(cdir,'mus.png'),dpi=300,bbox_inches='tight')
plt.cla()


plt.figure(figsize=(5,5))
plt.imshow(u2_all[-1,:,:], cmap='gray')
plt.axis('off')
plt.colorbar()
plt.savefig(os.path.join(cdir,f'reco.png'),dpi=300,bbox_inches='tight')
plt.cla()

plt.figure(figsize=(5,5))
plt.semilogy(S)
plt.savefig(os.path.join(cdir,f'svd.png'),dpi=300,bbox_inches='tight')
plt.cla()









