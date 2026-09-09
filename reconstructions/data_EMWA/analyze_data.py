import numpy as np

import matplotlib.pyplot as plt

import os
import site
relPathToMPI=os.path.join( os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir,os.pardir, 'utils' )
site.addsitedir(relPathToMPI)

#cmap = 'magma'
cmap = 'gray'
colorbar = False

cdir = os.path.dirname(__file__)
path2data = os.path.join(cdir,'reco_data.npz')

with open(path2data,'rb') as f:
    data = np.load(f)
    u1 = data['u1']
    u2 = data['u2']
    loss = data['loss']
    sigmas = data['sigmas']
    muit = data['mu_it']
    sv = data['sv']

# compute the PSNR
it_nums, Nx, Ny = u2.shape

for i in range(it_nums):

    tik = u1[i,:,:].T
    den = u2[i,:,:].T

    plt.figure(figsize=(5,5))
    plt.imshow(tik, cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(cdir,f'tik_{i}.png'),dpi=300,bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(5,5))
    plt.imshow(den, cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(cdir,f'den_{i}.png'),dpi=300,bbox_inches='tight')
    plt.clf()