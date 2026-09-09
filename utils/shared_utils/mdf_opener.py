import numpy as np
import torch

import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.set_default_device(device)

class DataOpener:

    def __init__(self,path2data=None):
        self.path2data = path2data

    def set_path2data(self,path):
        self.path2data = path

    def get_data(self):
        # open the data.mdf file
        data_mdf = h5py.File(self.path2data,'r')
        ################################
        # acquisition related parameters
        ################################
        self.gradient = data_mdf['/acquisition/gradient'][()].squeeze() # retrieve gradient matrix diag(-1,-1,2)
        self.numAverages = data_mdf['/acquisition/numAverages'][()] 
        self.numFrames = data_mdf['/acquisition/numFrames'][()]
        try:
            # what it should be in the Open MPI Dataset
            self.numPeriodsPerFrame = data_mdf['/acquisition/numPeriodsPerFrame'][()]
        except:
            # what it actually is...
            self.numPeriodsPerFrame = data_mdf['/acquisition/numPeriods'][()]

        try:
            self.offsetField = data_mdf['/acquisition/offsetField'][()]
        except:
            print('No offset field found! Might cause errors if is needed.')

        ################################
        # drivefield related parameters
        ################################
        self.baseFrequency = data_mdf['/acquisition/drivefield/baseFrequency'][()]   # base freq 2.5 MHz
        self.cycle = data_mdf['/acquisition/drivefield/cycle'][()]
        self.divider = data_mdf['/acquisition/drivefield/divider'][()] # frequency dividers (102, 96, 99)
        self.numChannels = data_mdf['/acquisition/drivefield/numChannels'][()] # num channels 3
        self.phase = data_mdf['/acquisition/drivefield/phase'][()].squeeze() # phase all pi/2 in forma (1,3,1)
        self.strength = data_mdf['/acquisition/drivefield/strength'][()].squeeze() # strenth diag (0.012, 0.012, 0) in format (1,3,1), [T/mu_0] 
        self.waveform = data_mdf['/acquisition/drivefield/waveform'][()] # waveform type, here sine
        
        if self.strength.ndim==1:
            self.amplitudes = self.strength/np.einsum('ii->i',self.gradient)  # strength [T/mu_0] divided by gradient [T/(m mu_0)] --> [m]
            self.amplitudes = self.amplitudes[np.newaxis,...]
            self.phase = self.phase[np.newaxis,...]
            self.strength = self.strength[np.newaxis,...]
        elif self.strength.ndim==2:
            self.amplitudes = self.strength/np.einsum('jii->ji',self.gradient)


        ################################
        # receiver info
        ################################
        self.receiverBandwidth = data_mdf['/acquisition/receiver/bandwidth'][()] # Receiver Bandwidth, here 1,250,000 Hz
        self.numSamplingPoints = data_mdf['/acquisition/receiver/numSamplingPoints'][()] # Nt = 1632
        try:
            self.transferFunction = data_mdf['/acquisition/receiver/transferfunction'][()] # complex valued array of shape (3,817)
        except:
            pass
            try:
                self.transferFunction = data_mdf['/acquisition/receiver/transferFunction'][()] # complex valued array of shape (3,817)
            except:
                print('There is no transfer function stored!')

        ################################
        # experiment related
        ################################
        self.data = data_mdf['/measurement/data'][()] # compl. v. matrix of shape (len_data, 1, 3, 817)

        self.isBackgroundCorrected = data_mdf['/measurement/isBackgroundCorrected'][()] # in this case 0, so no
        self.isBackgroundFrame = data_mdf['/measurement/isBackgroundFrame'][()] # in this case just a set of zeros
        self.isFastFrameAxis = data_mdf['/measurement/isFastFrameAxis'][()]  # Flag, if the frame dimension haveen moved tot he last dimension
        if self.isFastFrameAxis:
            self.data = np.rollaxis(self.data,-1,0)
        self.isFourierTransformed = data_mdf['/measurement/isFourierTransformed'][()] # int his case yes, so 1
        self.isFramePermutation = data_mdf['/measurement/isFramePermutation'][()] # this is the flag for the permutations of thye frame
        self.isFrequencySelection = data_mdf['/measurement/isFrequencySelection'][()] # falg for frequency selection
        
        try:
            # retrieve tracer information
            self.part_name = data_mdf['/tracer/name'][()] # perimag
            self.kanis = data_mdf['/tracer/_K_anis'][()] # [500 800 110 1400 1700 2000 2300 2600 2900 3200 3500 3800], 12 in total
            self.diameters = data_mdf['/tracer/_diameters'][()] # [16 18 20 22 24], 5 in total
            self.weights = data_mdf['/tracer/_weights'][()] # matrix 5x12
            self.avg_particle_diam = np.mean(self.diameters)*1e-9
        except:
            print("ATTENTION: one of the tracer info are not found!")


        
        # section only for SM data
        try:
            self.snr = data_mdf['/calibration/snr'][()]
            self.have_snr = True
        except:
            print('Nor calibration/snr data has been found')
            self.have_snr = False
        try:
            self.size = data_mdf['/calibration/size'][()]
            self.have_size = True
        except:
            print('Nor calibration/size data has been found')
            self.have_size = False
        try:
            self.fieldOfView = data_mdf['/calibration/fieldOfView'][()]
            self.have_fov = True
        except:
            print('Nor calibration/fieldOfViewdata has been found')
            self.have_fov = False
            

    def convert_to_time(self):

        fourier_signal = self.data

        # Perform the inverse FFT to get the time-domain signal
        time_domain_signal = np.fft.irfft(fourier_signal,axis=-1)

        # To get the real-valued signal, if necessary
        real_time_domain_signal = time_domain_signal.real

        self.data_timedomain = real_time_domain_signal
    
    def bandpass_filter(self,lower_limit=80000,upper_limit=625000):

        if upper_limit >= self.receiverBandwidth:
            upper_limit = None

        N = self.data.shape[-1]
        f_x = np.linspace(0,self.receiverBandwidth,N) # equidistant freqs up to store max freq in receiverBandWidth
        
        # create the mask for the frequencies between lower and upper limits
        if not (upper_limit==None):
            index_f_inband = np.where((f_x >= lower_limit) & (f_x <= upper_limit),False,True)
        elif upper_limit==None:
            index_f_inband = np.where(f_x >= lower_limit,False,True)


        new_data = self.data.copy()

        new_data[:,:,:,index_f_inband] = 0.0 + 0.0j

        self.data = new_data

    def bandpass_filter_axiswise(self,x=(-1,np.inf),y=(-1,np.inf),z=(-1,np.inf)):

        xyz = [x,y,z]

        new_data = self.data.copy()

        for ch in range(self.numChannels):
            
            lower_limit, upper_limit = xyz[ch]


            if upper_limit >= self.receiverBandwidth:
                upper_limit = None

            N = self.data.shape[-1]
            f_x = np.linspace(0,self.receiverBandwidth,N) # equidistant freqs up to store max freq in receiverBandWidth
            
            # create the mask for the frequencies between lower and upper limits
            if not (upper_limit==None):
                index_f_inband = np.where((f_x >= lower_limit) & (f_x <= upper_limit),False,True)
            elif upper_limit==None:
                index_f_inband = np.where(f_x >= lower_limit,False,True)


            new_data[:,:,ch,index_f_inband] = 0.0 + 0.0j

        self.data = new_data

    def use_transfer(self):
        self.data[:,:,0,:] /= self.transferFunction[0,:]
        self.data[:,:,1,:] /= self.transferFunction[1,:]
        self.data[:,:,2,:] /= self.transferFunction[2,:]

    def use_SNR(self, threshold=1):
        if not self.have_snr:
            raise Exception('No SNR, thresholding skipped!')
        
        mask_threshold = np.where(self.snr >= threshold, False, True)
        self.data[:,mask_threshold] = 0.0 + 0.0j

    def use_SNR_axiswise(self, x=0,y=0,z=0):
        if not self.have_snr:
            raise Exception('No SNR, thresholding skipped!')
        
        xyz = [x,y,z]


        for ch in range(self.numChannels):

            threshold = xyz[ch]
        
            mask_threshold = np.where(self.snr[0,ch,:] >= threshold, False, True)
            self.data[:,:,ch,mask_threshold] = 0.0 + 0.0j



    def reorder_separatebackground(self):

        # select background scans
        backgroundmask = self.isBackgroundFrame.astype(bool)
        foreground_mask = np.logical_not(self.isBackgroundFrame)
        self.background_data = self.data[backgroundmask,:,:,:]
        self.data = self.data[foreground_mask,:,:,:]

    def reorder_separatebackground2(self): # USED
        # select background scans
        backgroundmask = self.isBackgroundFrame.astype(bool)
        foreground_mask = np.logical_not(self.isBackgroundFrame)
        self.background_data = self.data[backgroundmask,:,:,:]
        self.data = self.data[foreground_mask,:,:,:]

    
    def torchify(self):

        self.data = torch.from_numpy(self.data).to(device)


    def stack(self):
        '''
        In this function we stack the imaginary and real parts as well as the channels
        to create a unique system matrix.
        '''

        # first we stack the real and imaginar parts
        realdata = self.data.real
        imdata = self.data.imag

        self.data = torch.cat((realdata,imdata),dim=-1).squeeze()

        # then we stack each channel
        self.data = torch.cat((self.data[:,0,:],self.data[:,1,:]),dim=1)

    def rsvd(self,nrows = None):
        '''
        With this function we perform the randomized SVD.
        As an input we take from the descriptor the rank K needed.
        If this rank K is bigger than the full rank, than the full SVD is computed
        but a warning message is thrown!

        The rSVd algorihtm is torch.svd_lowrank based on the 
        Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding structure with randomness: 
        probabilistic algorithms for constructing approximate matrix decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009.

        We also take sedd fro reproducibility from the descriptor.
        '''
        if nrows != None:
            A = self.data

            K = nrows

            # check is K bigger than full rank
            full_rank = min(A.shape[1], A.shape[0])
            if K >= full_rank:
                print(f'The rank chosen is bigger than the full rank: SVD with rank {full_rank} will be computed.')
                K = full_rank

            # perform the (reduced) svd A = U * diagS * V^T
            self.U,self.S,self.V = torch.svd_lowrank(A,q=K) 
            self.Vh = self.V.adjoint()
