import numpy as np
import torch

print(torch.__version__)

import shutil
import sys
import subprocess

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

phantom_name = 'spiral'  # choose between : 'spiral', 'resolution1', 'resolution2', 'resolution3', 'icecream', 'dot'
upscale_const = 3        # upscaling factor for the super-resolution. 1 corresponds to no upscaling.
mode = 'bilinear'        # choose between : 'bilinear', 'repeat' . 'repeat' stands for nearest neighbors

translate_phantom = {
    'spiral' : '2',
    'resolution3' : '3',
    'resolution2' : '4',
    'resolution1' : '5',
    'icecream' : '6',
    'dot' : '7',
}

#### define paths to import the data
cdir = os.path.dirname(__file__)
path2data = os.path.join(cdir,os.pardir,os.pardir,"datasets",'EMWAdata')
path2measurements = os.path.join(path2data,'measurements','20230613_150948_model based fluid')
path2calibration = os.path.join(path2data,'calibrations')


path2meas = os.path.join(path2measurements,translate_phantom[phantom_name]+'.mdf')   # measurements
path2bg = os.path.join(path2measurements,'1.mdf')                                    # background scans
path2sm = os.path.join(path2calibration,'13.mdf')                                    # calibration data (SM)


# extract the system matrix
sm = DataOpener()
sm.set_path2data(path2sm)
sm.get_data()    # data shape (811, 1, 3, 817)
sm.reorder_separatebackground2() # data shape (756,1,3,817)
sm.data -= np.mean(sm.background_data,axis=0)  # background (46,1,3,817) -> (1,1,3,817) and then subtracted

# extract the background scans
bg = DataOpener()
bg.set_path2data(path2bg)
bg.get_data() # .data.shape = (20,1,3,1632)
bg.transferFunction = bg.transferFunction 
# background correction average
bg.data = np.mean(bg.data,axis=0)[np.newaxis,:,:,:] # .data.shape = (20,1,3,817) -> (1,1,3,1632)
bg.data = np.fft.rfft(bg.data,axis=-1) # .data.shape = (1,1,3,1632) -> (1,1,3,817)



grid_full = sm.size
Ny,Nx,_ = grid_full
grid = (Nx,Ny)
voxel_num = Nx*Ny

# descriptor of the experiments

desc = {
        'bandpass_lower'     : 80000,           # high pass at 80 kHz
        'bandpass_upper'     : np.inf,          # low pass at 625 kHz or high frequency filter if set to np.inf
        'SNR_threshold'      : 1,               # SNR thresholding of the frequencies at SNR level 1
        'whitening'          : False,           # will whiten from calibration scan ()
        'rSVD'               : Nx*Ny//2,        # target rank of the randomized SVD
        'seed'               : 145,             # reproducibility seed
        # reconstruction variables
        'mu_0'               : 1e9,             # starting parameter mu_0 of the algorithm
        'pnp_iter'           : 10,              # number of plug-and-play iterations
        'noise_estimation'   : 'variance',      # currently the only available option
        # CG related parameters
        'cg_tolerance'       : 1e-12,            # tolerance for the Conjugatedd Gradient methods
        'cg_maxit'           : 1000,             # maximal iteration number for the CG method
    }



##################################################################################
################### Extracxt the data ############################################
##################################################################################

sm.data = sm.data[-voxel_num:,...] # new shape of sm.data is (17x15 , 1, 3, 817)   , 17x15 = 255
sm.use_SNR(threshold=desc['SNR_threshold'])
sm.bandpass_filter(lower_limit=desc['bandpass_lower'],upper_limit=desc['bandpass_upper'])
sm.torchify()
sm.stack()
sm.data = sm.data.to(type)
sm.data = sm.data.T
sm.rsvd(desc['rSVD'])
U,S,V,Vh = sm.U, sm.S, sm.V,sm.Vh


# extract the data measurement
noisy_data = DataOpener()
noisy_data.set_path2data(path2meas)
noisy_data.get_data() # .data.shape = (100,1,3,1632)
noisy_data.reorder_separatebackground2()
noisy_data.snr = sm.snr  # snr estimation from external source
noisy_data.have_snr = True
noisy_data.data = np.fft.rfft(noisy_data.data,axis=-1)  # new data shape (100,1,3,817)
noisy_data.data = noisy_data.data - bg.data
noisy_data.use_SNR(threshold=desc['SNR_threshold'])
noisy_data.bandpass_filter(lower_limit=desc['bandpass_lower'],upper_limit=desc['bandpass_upper'])
noisy_data.torchify()
noisy_data.stack()   # data shape (100, 3268) and SM.sahpe = (3268, 255)
noisy_data.data = noisy_data.data.to(type)

# select first exp_num expertiments and store them
# use the U matrix to transform all data
new_data = torch.einsum('ni,ij->nj',noisy_data.data,U)
scan = new_data[0,...]


### create storage folder and dictionary
cdir = os.path.dirname(__file__)
path2save = os.path.join(cdir,'reconstruction_results',f'{phantom_name}',f'rec_{Nx}x{Ny}_up_{upscale_const}')
os.makedirs(path2save,exist_ok=True)

reco_obj = recoPNP()
reco_obj.setup_model()
reco_obj.u = scan
Vh = V.adjoint()
reco_obj.A = S[:, None] * Vh
reco_obj.Nx, reco_obj.Ny = Nx, Ny


## reconstruction method

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