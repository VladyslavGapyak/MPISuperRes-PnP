import numpy as np
import torch

import shutil
import subprocess

import sys

import os
import site
relPathToMPI=os.path.join( os.path.dirname(__file__), os.pardir,os.pardir,'utils')
site.addsitedir(relPathToMPI)

from cldef_pnp import recoPNP
from mdf_opener import DataOpener

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
type = torch.float64
torch.set_default_device(dev)
torch.set_default_dtype(type)

#### data import and preprocessing
phantom_name = 'shape'
upscale_const = 3
mode = 'bilinear'

##### Define paths
cdir = os.path.dirname(__file__)
path2data = os.path.join(cdir,os.pardir,os.pardir,"datasets",'openmpi')

path2meas = os.path.join(path2data,'measurement',phantom_name+'Phantom','2.mdf')          # scan
path2sm = os.path.join(path2data,'calibration','2.mdf')    # calibration

grid = (19,19)
Nx, Ny = grid
# descriptor of the experiments


desc = {
        'bandpass_lower'     : 80000,           # high pass at 80 kHz
        'bandpass_upper'     : np.inf,          # low pass at 625 kHz or high frequency filter if set to np.inf
        'SNR_threshold'      : 0,               # SNR thresholding of the frequencies at SNR level 1
        'whitening'          : False,           # will whiten from calibration scan ()
        'rSVD'               : 19*19//2,             # target rank of the randomized SVD
        'seed'               : 145,             # reproducibility seed
        'extract_submatrix'  : False,
        # reconstruction variables
        'mu_0'               : 1e8,             # starting parameter mu_0 of the algorithm
        'pnp_iter'           : 10,              # number of plug-and-play iterations
        'noise_estimation'   : 'variance',      # currently the only available option
        # CG related parameters
        'cg_tolerance'       : 1e-12,            # tolerance for the Conjugatedd Gradient methods
        'cg_maxit'           : 1000,             # maximal iteration number for the CG method
    }




##################################################################################
################### Extracxt the data ############################################
##################################################################################
j=11
exp_num_low = (j-1)*1000 + 1
exp_num_up = j*1000
slice_z = 9
# for resolution and shape 9 is ok

# extract the system matrix reconstruction
sm = DataOpener()
sm.set_path2data(path2sm)
sm.get_data()    # data shape (6375, 1, 3, 817)
sm.reorder_separatebackground2()
sm.data -= np.mean(sm.background_data,axis=0)
plane_size = Nx*Ny
start = slice_z*plane_size
end = start + plane_size
sm.data = sm.data[start:end,:,:,:]
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
noisy_data.set_path2data(path2meas)
noisy_data.get_data() # .data.shape = (10000,1,3,817)
noisy_data.data = np.fft.rfft(noisy_data.data,axis=-1)
noisy_data.reorder_separatebackground()
noisy_data.snr = sm.snr  # snr estimation from external source
noisy_data.have_snr = True
noisy_data.data = noisy_data.data - np.mean(noisy_data.background_data,axis=1)[:,np.newaxis,:,:]
noisy_data.data = np.swapaxes(noisy_data.data,0,1)
noisy_data.use_SNR(threshold=desc['SNR_threshold'])
noisy_data.bandpass_filter(lower_limit=desc['bandpass_lower'],upper_limit=desc['bandpass_upper'])
noisy_data.torchify()
noisy_data.stack()
noisy_data.data = noisy_data.data.to(type)

# select first exp_num expertiments and store them
# use the U matrix to transform all data
new_data = torch.einsum('ni,ij->nj',noisy_data.data,U)
scan = new_data[exp_num_low:exp_num_up,...]
scan = torch.mean(scan,dim=0)


### create storage folder and dictionary
cdir = os.path.dirname(__file__)
path2save = os.path.join(cdir,'reconstruction_results',f'{phantom_name}_{mode}_{upscale_const}')
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



with open(os.path.join(path2save,"reco_data.npz"),"wb") as f:
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