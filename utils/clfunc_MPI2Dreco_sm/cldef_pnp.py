import torch
import torch.nn.functional as F
import numpy as np

import os
import math
import time
from skimage.restoration import estimate_sigma



currentdir = os.path.dirname(__file__)

from cldef_cg_torch import LinearOperator_torch
from cldef_upscaler import UpscaleOperator

import sys
sys.path.append(os.path.join(currentdir,'models'))
from unet import UNetRes as net

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {device}')


class recoPNP:

    def __init__(self):
        self.SM = None
        self.y = None 

        self.model = None
        self.model_name = None

    def pad(self,x,pad_width):
        """
        input: 
        - x: np array
        - pad_width: integer or float
        output:
        - x_pad: np array
        """
        x_pad = np.pad(x, (math.floor(pad_width/2),math.ceil(pad_width/2)),mode='edge')
        return x_pad
    
    def pad_torch(self,x,pad_width):
        """
        input: 
        - x: np array
        - pad_width: integer or float
        output:
        - x_pad: np array
        """
        x_pad = torch.nn.functional.pad(x, (math.floor(pad_width[1]/2),math.ceil(pad_width[1]/2),math.floor(pad_width[0]/2),math.ceil(pad_width[0]/2)),mode='constant',value=0)
        return x_pad
    
    def unpad(self,x,pad_width):
        x_unpadded = x[math.floor(pad_width[0]/2):-math.ceil(pad_width[0]/2),math.floor(pad_width[1]/2):-math.ceil(pad_width[1]/2)]
        return x_unpadded
    
    def unpad_torch(self,x,pad_width):
        x_unpadded = x[math.floor(pad_width/2):-math.ceil(pad_width/2),math.floor(pad_width/2):-math.ceil(pad_width/2)]
        return x_unpadded
    
    def sigma_laplacian_mad(self,x):
        # x: HxW, CxHxW, or NxCxHxW (float, any range)
        if x.dim() == 2: x = x[None,None]
        elif x.dim() == 3: x = x[None]
        k = torch.tensor([[1,-2,1],[-2,4,-2],[1,-2,1]], device=x.device, dtype=x.dtype)/4.0
        k = k.view(1,1,3,3).repeat(x.shape[1],1,1,1)
        r = F.conv2d(F.pad(x,(1,1,1,1),mode="reflect"), k, groups=x.shape[1])
        r = r.reshape(r.shape[0], -1)
        mad = (r - r.median(dim=1, keepdim=True).values).abs().median(dim=1).values
        return (1.4826 * mad / 1.5).squeeze()  # 1.5 ≈ ||k||_2
    
    def denoise_via_model_torch_up(self,x_cube,model,noiselevel_const,slice_axis=2):# USED
        """  
        denoise a reconstructed MPI result using a net
        inputs
        - x_cube: reconstruction in a cube shape lxmxn
        - model: network
        - noiselevel_const: constant noiselevel
        - slice_axis: axis, along which <x_cube> is sliced into images
        output
        - denoised result of <x_cube>, has the same shape like <x_cube>
        """
        Nx = self.up_times * self.Nx
        Ny = self.up_times * self.Ny
        pad_width = ( 2**(int(np.log2(Nx))+1 ) - Nx   ,   2**(int(np.log2(Ny))+1 ) - Ny )
        
        # pad slice in order to fit the model
        x_slice = self.pad_torch(x_cube,pad_width)

        if self.model_name=='zhang':

            # expand the image shape from HxW to HxWx1 to prepare concatenation with constant noise level map
            z = x_slice.unsqueeze(0).unsqueeze(0)

            # concatenate the image with a noise level map (according to FFDNet (Zhang, et al.)) -> shape: torch.Size([1, 2, H, W])
            z = torch.cat((z, noiselevel_const.float().repeat(1, 1, z.shape[2], z.shape[3])), dim=1)


            # denoise using the model
            op = model(z)#.double()
            #op_np_sq = op.cpu().detach().numpy().squeeze()
            op_np_sq = op.squeeze()

            # unpad result
            op_np_sq_up = self.unpad(op_np_sq, pad_width)
            

        return op_np_sq_up
    
    def setup_model(self):
        # preparation for denoising via model
        n_channels = 1  # number of channels is 1 for MPI using one Tracer material

        # define paths of pretrained model
        pretrained_model_dir = os.path.join(currentdir,'models')
        pretrained_model_name = 'drunet_gray.pth'
        pretrained_model_path = os.path.join(pretrained_model_dir, pretrained_model_name)

        torch.cuda.empty_cache()

        # initialize model
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4)
        # load pretrained model
        model.load_state_dict(torch.load(pretrained_model_path), strict=True)
        # use evaluation mode
        model.eval()
        # turn off the computation of the gradients
        for _, params in model.named_parameters():
            params.requires_grad = False
        # model computations are stored in device (gpu or cpu)
        model = model.to(device)

        self.model = model

        self.model_name = 'zhang'
    
    
    def tikhonov_operator(self,b=None):
        '''
        Defines the operator A^*A + mu * U* U used in the CG iterations, where
        U is the upscale function.
        '''
        atax = self.AtA @ b 

        if self.up_times == 1:
            return atax + self.mu_k * b
        else:
            b_square = b.reshape((1,1,self.Nx,self.Ny))
            convo = self.upscaler.UTU(b_square).ravel()
            return atax + self.mu_k * convo

    def pnp_HQS_upscale(self,mu_0 = 1, up_times = 1, mode = 'repeat', iter = 10, denoiser="our",noise_estim = 'variance',cg_maxit = 1000, cg_tol = 1e-12):# USED
        # PnP main

        self.up_times = up_times
        self.time_collector = []

        # include the upscaling object
        self.upscaler = UpscaleOperator(s=up_times, mode=mode)

        start_time = time.perf_counter_ns()
        self.time_collector.append(start_time)
    
        # hyperparam collectors lists
        mus = [mu_0]
        sigmas = [torch.inf]

        # load data
        y = self.u
        A = self.A

        gridsize = A.shape[1]

        self.Atf = torch.einsum('ji,j->i',A,y)
        self.AtA = torch.einsum('ji,jk->ik',A,A)

        # create dummy variables
        u1_all_vec = [torch.zeros(gridsize,device=device)]
        u1_all_2d = [torch.zeros(self.Nx,self.Ny,device=device)]
        u2_all_vec = [torch.zeros(up_times**2 * gridsize,device=device)]
        u2_all_2d = [torch.zeros(up_times*self.Nx,up_times*self.Ny,device=device)]

        loss_all = [torch.inf]

        self.cg_conv = []
        self.cg_its  = []

        for iteration in range(iter):
            # reconstruction of the first stage, tikhonov with parameter mu_iteration
            #     ( A^*A + mu_k * U^* U ) x  =  A^* data + mu_k * U^* iterate
            mu_k = mus[-1]
            self.mu_k = mu_k

            # reconstruction with TIkhonov
            iterate = u2_all_vec[-1]
            iterate_square = iterate.reshape((1,1,up_times*self.Nx,up_times*self.Ny))
            
            lefthand_side = LinearOperator_torch(self.AtA.shape, matvec = self.tikhonov_operator)
            if up_times == 1:
                UTu2 = iterate
            else:
                UTu2 = self.upscaler.UT(iterate_square).ravel()
            righthand_side = self.Atf + mu_k * UTu2

            u1, conv, cgit = lefthand_side.cg(righthand_side,x0=UTu2,maxit=cg_maxit,tol=cg_tol)

            # add convergence info
            self.cg_conv.append(conv)
            self.cg_its.append(cgit)

            
            # add reconstruction to the list of u1 s
            u1_all_vec.append(u1)
            
            u1_square = torch.reshape(u1, (self.Nx,self.Ny))
            u1_all_2d.append(u1_square)

            # estimate noise level of current u1
            if noise_estim=="variance":
                noise_level = torch.sqrt(torch.mean((u1-u1.mean())**2))
                sigmas.append(noise_level.clone().detach().cpu().numpy().item())
            elif noise_estim=="MAD":
                input_estimate = u1_square.clone().detach().cpu().numpy()
                noise_level = estimate_sigma(input_estimate)
                sigmas.append(noise_level)
                noise_level = torch.tensor(noise_level)
            elif noise_estim=="LaplacianMAD":
                noise_level = self.sigma_laplacian_mad(u1_square)
                sigmas.append(noise_level.clone().detach().cpu().numpy().item())

            if iteration==0:
                lamb = mu_k*sigmas[-1]**2

            # denoising
            if up_times == 1:
                u1_upscale = u1_square
            else:
                u1_upscale = self.upscaler.U(u1_square.unsqueeze(0).unsqueeze(0))


            u2_square = self.denoise_via_model_torch_up(u1_upscale,model=self.model,noiselevel_const=noise_level,slice_axis=0)

            u2_all_2d.append(u2_square)

            u2_vec = torch.reshape(u2_square, (up_times**2 * self.Nx*self.Ny,))
            u2_all_vec.append(u2_vec)

            
            loss_tmp = ((u2_square[-1] - u2_square[-2]) ** 2).mean()
            loss_all.append(loss_tmp)

            mus.append(lamb/sigmas[-1]**2)

            end_time = time.perf_counter_ns()
            self.time_collector.append(end_time)

            print(f'[IT] : {iteration}/{iter} [Rec. param] : {mu_k} [Noise pam. sigma_k] : {sigmas[-1]**2}')

        return u1_all_2d, u2_all_2d, mus, sigmas, loss_all