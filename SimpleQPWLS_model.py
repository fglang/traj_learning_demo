
'''
Joint optimization of sampling trajectory with QPLS/CG-SENSE reconstruction
'''
import torch
import itertools
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from util.metrics import PSNR
# import pytorch_msssim
import time
import sys
import os
import numpy as np
from util.hfen import hfen

import scipy.io as sio

from mirtorch.alg import CG, FISTA, POGM, power_iter, FBPD
from mirtorch.linear import LinearMap, FFTCn, NuSense, Sense, FFTCn, Identity, Diff2dgram, Diff3dgram, Gmri, \
    Wavelet2D, \
    NuSenseGram, Diffnd
from mirtorch.prox import Prox, L1Regularizer, Const
from models.mirtorch_pkg import NuSense_om, Gram_inv, NuSenseGram_om, Gram_inv_diff
import torchkbnufft as tkbn
from models.losses import pns
from models.sgld import SGLD

import matplotlib.pyplot as plt


class SimpleQPWLSModel(BaseModel):
    def name(self):
        return 'SimpleQPWLSModel'

    # Initialize the model
    def initialize(self, opt):
        BaseModel.initialize(self, opt)  # ATTENTION HERE: NEED TO ALTER THE DEFAULT PLAN
        # Define the parameterization strategy. Please replace you own here.
        self.netSampling = networks.define_G(opt, opt.which_model_netD_I, opt.init_type, opt.init_gain, self.gpu_ids)
        # Define the terms
        if self.isTrain:
            self.model_names = ['Sampling']
            self.loss_names = ['G_I_L1', 'G_I_L2', 'grad', 'slew']

        else:  # during test time, only load Gs
            self.model_names = ['Sampling']
            self.loss_names = ['PSNR']

        # Define the visual terms in the Visdom
        self.visual_names = ['Ireal', 'Ifake', 'Iunder', 'ktraj', 'Idcf', 'grad', 'slew', 'pt']

        self.num_shots = opt.num_shots
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        # Choose which optimizer to use
        if self.isTrain:
            self.optimizers = []
            if self.opt.sgld:
                self.optimizer_S = SGLD(list(
                    self.netSampling.parameters()), lr=self.opt.ReconVSTraj * opt.lr_traj, num_burn_in_steps = 0)
            else:
                self.optimizer_S = torch.optim.Adam(list(
                    self.netSampling.parameters()), lr=self.opt.ReconVSTraj * opt.lr_traj, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S)
        self.update_alt_cnt = 0
        # Penalty on the TE locations.
        if self.opt.contrast_condition is not None:
            self.contrast_idx = np.load(self.opt.contrast_condition)['idx']
            self.contrast_value = np.load(self.opt.contrast_condition)['value']
            self.gradient_idx = np.load(self.opt.contrast_condition)['gidx']
            self.gradient_value = np.load(self.opt.contrast_condition)['gvalue']

    def set_input(self, fname_input):
        input = sio.loadmat(fname_input)
        self.Ireal = torch.tensor(input['I'], dtype=torch.cfloat).to(self.device)
        self.smap = torch.tensor(input['S'], dtype=torch.cfloat).to(self.device)

        # FG: shape adjustments for forward operators
        self.smap = self.smap.unsqueeze(0)
        self.Ireal = self.Ireal.unsqueeze(0).unsqueeze(0)


    def backward_G(self):
        # Define the loss function
        self.loss_G_I_L1 = self.criterionL1(self.Ifake, self.Ireal) * self.opt.loss_content_I_l1
        self.loss_G_I_L2 = self.criterionMSE(self.Ifake, self.Ireal) * self.opt.loss_content_I_l2
        self.loss_G_CON_I = self.loss_G_I_L1 + self.loss_G_I_L2
        softgrad = torch.nn.Softshrink(self.opt.gradmax * 0.995)
        softslew = torch.nn.Softshrink(self.opt.slewmax * 0.995)
        softpi = torch.nn.Softshrink(torch.pi)
        pt = torch.norm(self.pt, dim=0)

        # FG: ignore PNS loss in this simple example
        # softpns = torch.nn.Softshrink(self.opt.pth)
        # self.loss_pns = torch.sum(softpns(pt))*self.opt.loss_pns

        if self.opt.iso_constraint:
            self.loss_grad = torch.sum(torch.pow(softgrad(torch.norm(self.grad, dim=0)), 2))*self.opt.loss_grad
            self.loss_slew = torch.sum(torch.pow(softslew(torch.norm(self.slew, dim=0)), 2))*self.opt.loss_slew
        else:
            self.loss_grad = torch.sum(torch.pow(softgrad(torch.abs(self.grad)), 2))*self.opt.loss_grad
            self.loss_slew = torch.sum(torch.pow(softslew(torch.abs(self.slew)), 2))*self.opt.loss_slew
        
        self.loss_G = self.loss_G_CON_I + self.loss_grad + self.loss_slew 

        self.loss_G.backward()

    def forward(self):
        self.ktraj, self.grad, self.slew = self.netSampling(1) # get trajectory according to current parameters
        
        if self.opt.fix_first_k0:
            self.ktraj = self.ktraj.reshape(3, self.opt.num_shots, -1)
            self.ktraj[:,:,0] = 0
            self.ktraj = self.ktraj.reshape(1,3,-1)
            
        self.ktraj = self.ktraj[:,:-1,:] # FG: ignore z-component for 2D
        self.pt = pns(self.slew)

        # smaps: sensitivity maps in [nbatch, nc, nx, ny, (nz)]
        # traj: sampling trajectory in size [dim, npoints]
        self.A = NuSense_om(self.smap, self.ktraj, numpoints=self.opt.numpoints, grid_size=self.opt.grid_size,
                            norm='ortho')
        self.kunder = self.A * (self.Ireal)
        # Simulate the additive Gaussian noise
        self.kunder = self.kunder + self.opt.noise_level*torch.randn_like(self.kunder)
        self.Iunder = self.A.H * self.kunder
        self.dcf = tkbn.calc_density_compensation_function(self.ktraj.detach(), im_size=self.Ireal.shape[2:])
        self.Idcf = self.A.H * (self.kunder * self.dcf).detach()
        AIdcf = self.A * self.Idcf
        self.Idcf = self.Idcf * torch.sum(torch.conj(AIdcf) * self.kunder) / (torch.norm(AIdcf) ** 2)
        # Decide whether to use QPWLS or CG-SENSE.
        if self.opt.use_rough:
            self.P = Gram_inv_diff(self.smap, self.ktraj, self.opt.CGlambda, self.opt.CGtol,
                                    max_iter=self.opt.num_blocks, norm='ortho',
                                    numpoints=self.opt.numpoints, grid_size=self.opt.grid_size, alert=False,
                                    x0=self.Idcf)
            self.Ifake = self.P * self.Iunder
        else:
            self.P = Gram_inv(self.smap, self.ktraj, self.opt.CGlambda, self.opt.CGtol,
                               max_iter=self.opt.num_blocks, norm='ortho',
                               numpoints=self.opt.numpoints, grid_size=self.opt.grid_size, alert=False, x0=self.Idcf)
            self.Ifake = self.P * self.Iunder
        self.Ireal = torch.view_as_real(self.Ireal.squeeze(1).unsqueeze(-1)).permute(0, 4, 1, 2, 3) # FG: complex 2D representation as dim1
        self.Ifake = torch.view_as_real(self.Ifake.squeeze(1).unsqueeze(-1)).permute(0, 4, 1, 2, 3)
        self.Idcf = torch.view_as_real(self.Idcf.squeeze(1).unsqueeze(-1)).permute(0, 4, 1, 2, 3)
        self.Iunder = torch.view_as_real(self.Iunder.squeeze(1).unsqueeze(-1)).permute(0, 4, 1, 2, 3)
        self.loss_PSNR = PSNR(self.Ireal, self.Ifake)

    def optimize_parameters(self):
        start = time.time()
        self.forward()
        self.optimizer_S.zero_grad()
        self.backward_G()
        self.optimizer_S.step()
        print(f'Epoch took {time.time() - start} s')
