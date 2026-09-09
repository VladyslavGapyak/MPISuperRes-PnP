
import numpy as np
import torch
import h5py

import matplotlib.pyplot as plt

import shutil
import subprocess
import sys

import os
import site
relPathToMPI=os.path.join( os.path.dirname(__file__), os.pardir,os.pardir,'utils')
site.addsitedir(relPathToMPI)

from cldef_pnp import recoPNP
from mdf_opener_mnist import DataOpener

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type = torch.float64
torch.set_default_device(dev)
torch.set_default_dtype(type)

#### data import and preprocessing
##### Define paths
cdir = os.path.dirname(__file__)
path2data = os.path.join(cdir,os.pardir,os.pardir,"datasets",'mpimnist')

path2gts = os.path.join(path2data,'test_gt','test_gt.hdf5')                       # ground truths
path2noisy = os.path.join(path2data,'test_obsnoisy','test_obsnoisy.mdf')          # noisy scans
#path2noisy = os.path.join(path2data,'test_obs','test_obs.mdf')                  # not noisy scans
path2bg = os.path.join(path2data,'test_noise','NoiseMeas_phantom_bg_test.mdf')    # noise samples for background correction of noise perturbed MPI measurements 

sm_type = 'coarse' # 'coarse' 'int','fine'
mode = 'bilinear'
upscale_const = 3

path2sm = os.path.join(path2data,'SM','SM_fluid_small_params_'+sm_type+'.mdf')           # system matrix for the reconstruction
#path2sm = os.path.join(path2data,'SM','SM_fluid_equilibrium_'+sm_type+'.mdf')           # system matrix for the reconstruction
path2sm = os.path.join(path2data,'SM','SM_fluid_opt_'+sm_type+'.mdf')           # system matrix for the reconstruction
smtype2grid = {
    'coarse' : (15,17),
    'int'    : (45,51),
    'fine'   : (75,85),
    }

grid = smtype2grid[sm_type]
Nx, Ny = grid


# descriptor of the experiments, modify it for the preprocessing and for the.

desc = {
        'bandpass_lower'     : 77000,           # high pass at 80 kHz
        'bandpass_upper'     : np.inf,          # low pass at 625 kHz or high frequency filter if set to np.inf
        'SNR_threshold'      : 0,               # SNR thresholding of the frequencies at SNR level 1
        'whitening'          : False,           # will whiten from calibration scan ()
        'rSVD'               : 100,             # target rank of the randomized SVD
        'seed'               : 145,             # reproducibility seed
        'extract_submatrix'  : False,
        # reconstruction variables
        'mu_0'               : 1e-10,             # starting parameter mu_0 of the algorithm
        'pnp_iter'           : 10,              # number of plug-and-play iterations
        'noise_estimation'   : 'variance',      # currently the only available option
        # CG related parameters
        'cg_tolerance'       : 1e-12,            # tolerance for the Conjugatedd Gradient methods
        'cg_maxit'           : 1000,             # maximal iteration number for the CG method
    }




##################################################################################
################### Extracxt the data ############################################
##################################################################################
exp_num = 0

gts = h5py.File(path2gts,'r')['phantom_data'][()] # shape (10000, 15x17)

# extract background noise
noise = DataOpener()
noise.set_path2data(path2bg)
noise.get_data()    # data shape (100, 1, 3, 817)
noisemean = np.mean(noise.data,axis=0)[np.newaxis,:,:,:]
noise.data = noise.data[0,:,:,:][np.newaxis,:,:,:]  # (1,1,3,817)

# extract the system matrix reconstruction
sm = DataOpener()
sm.set_path2data(path2sm)
sm.get_data()    # data shape (6375, 1, 3, 817)
#sm.reorder_separatebackground()
sm.data -= noisemean
sm.use_SNR(threshold=desc['SNR_threshold'])
sm.bandpass_filter(lower_limit=desc['bandpass_lower'],upper_limit=desc['bandpass_upper'])
sm.torchify()
sm.stack()
sm.data = sm.data.T
sm.data = sm.data.to(type)
sm.rsvd(desc['rSVD'])
U,S,V = sm.U, sm.S, sm.V


# extract the data measurement
noisy_data = DataOpener()
noisy_data.set_path2data(path2noisy)
noisy_data.get_data() # .data.shape = (10000,1,3,817)
#noisy_data.reorder_separatebackground()
noisy_data.snr = sm.snr  # snr estimation from external source
noisy_data.have_snr = True
noisy_data.data = noisy_data.data - noisemean
noisy_data.use_SNR(threshold=desc['SNR_threshold'])
noisy_data.bandpass_filter(lower_limit=desc['bandpass_lower'],upper_limit=desc['bandpass_upper'])
noisy_data.torchify()
noisy_data.stack()
noisy_data.data = noisy_data.data.to(type)

# select first exp_num expertiments and store them
# use the U matrix to transform all data
new_data = torch.einsum('ni,ij->nj',noisy_data.data,U)
scan = new_data[exp_num,...]


### create storage folder and dictionary
cdir = os.path.dirname(__file__)
path2save = os.path.join(cdir,'reconstruction_results',f'recPnP_{mode}_{upscale_const}',f'grid_{sm_type}')
os.makedirs(path2save,exist_ok=True)

reco_obj = recoPNP()
reco_obj.setup_model()
reco_obj.u = scan
Vh = V.adjoint()
reco_obj.A = S[:, None] * Vh
reco_obj.Nx, reco_obj.Ny = Nx, Ny
# fine 1e-3, coarse 1e-10
u1_all, u2_all, mus, sigmas, loss = reco_obj.pnp_HQS_upscale(mu_0 = desc['mu_0'], 
                                                             up_times = upscale_const, 
                                                             mode = mode, 
                                                             iter = desc['pnp_iter'], 
                                                             noise_estim = desc['noise_estimation'],
                                                             cg_maxit = desc['cg_maxit'],
                                                             cg_tol = desc['cg_tolerance'])

u1_all = [reco.detach().cpu().numpy() for reco in u1_all]
u2_all = [reco.detach().cpu().numpy() for reco in u2_all]
loss = [loss[0]] + [i.cpu() for i in loss[1:]]



with open(os.path.join(path2save,"recos.npz"),"wb") as f:
    np.savez(f,
             u1 = np.array(u1_all),
             u2 = np.array(u2_all),
             loss = np.array(loss),
             sigmas = np.array(sigmas),
             mu_it = np.array(mus),
             sv = np.array(sm.S.detach().cpu())
             )
    
### copy analysis file into the new folder
file_name = 'analyze_data.py'
path2analyze = os.path.join(path2save,file_name)
shutil.copy(os.path.join(cdir,file_name), path2save)

subprocess.run([sys.executable, path2analyze])

gt = torch.tensor(gts[exp_num,:].reshape((1,1,15,17)))
gt = reco_obj.upscaler.U(gt).squeeze().cpu().numpy()
plt.figure()
plt.imshow(gt,cmap='gray')
plt.savefig(os.path.join(path2save,'gt.png'),dpi=300,bbox_inches='tight')