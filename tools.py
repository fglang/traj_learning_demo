import torch
import numpy as np
import pypulseq as pp
import matplotlib.pyplot as plt

gamma_ = 42.5764 # MHz/T

def plot_traj(ktraj, num_shots, nfe):
    ''' plot of k-space trajectory '''

    traj_temp = ktraj.reshape([2,num_shots,nfe]).detach().cpu()
    for ishot in range(num_shots):
        plt.plot(traj_temp[0,ishot,:], traj_temp[1,ishot,:], '.-')
    plt.title('trajectory')
    plt.xlabel('kx'), plt.ylabel('ky')
    plt.xlim([-3.5,+3.5])
    plt.ylim([-3.5,+3.5])


def plot_epoch(model, opt, epoch, do_save=False, logloss=True):
    ''' plot k-space trajectory and images at every epoch'''
    ### k-space trajectory
    plt.figure(111, figsize=(20,10))
    plt.subplot(2,4,1)
    plot_traj(model.ktraj, opt.num_shots, opt.nfe)

    ### images
    Ifake_mag = torch.view_as_complex(model.Ifake.squeeze().permute([1,2,0])).abs()
    Iunder_mag = torch.view_as_complex(model.Iunder.squeeze().permute([1,2,0])).abs()
    Ireal_mag = torch.view_as_complex(model.Ireal.squeeze().permute([1,2,0])).abs()
    diff_img = Ireal_mag - Ifake_mag

    plt.subplot(2,4,2)
    plt.imshow(Ireal_mag.cpu().detach(), vmin=0, vmax=Ireal_mag.max(), cmap='gray')
    plt.colorbar()
    plt.title('ground truth')

    plt.subplot(2,4,3)
    plt.imshow(Ifake_mag.cpu().detach(), vmin=0, vmax=Ireal_mag.max(), cmap='gray')
    plt.colorbar()
    plt.title('CG SENSE')

    plt.subplot(2,4,4)
    plt.imshow(diff_img.cpu().detach(), vmin=-0.2, vmax=0.2)
    plt.colorbar()
    plt.title('diff: ground truth - CG SENSE')

    ### loss curves
    plt.subplot(2,3,4)
    for loss_name in model.loss_names:
        if logloss:
            plotfunc = plt.semilogy
        else:
            plotfunc = plt.plot
        plotfunc(model.all_losses[loss_name], '.-', label=loss_name)
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.title('losses')

    ### gradient waveforms and slew rates
    # factors 10 and 100 are used for rescaling between Gauss, Tesla, m, cm, ...
    plt.subplot(2,3,5)
    plt.plot(model.grad[0,:,:].flatten().detach().cpu()*10, '.-')
    plt.plot(model.grad[1,:,:].flatten().detach().cpu()*10, '.-')
    xl = torch.tensor(plt.xlim())
    plt.plot(xl, xl*0+opt.gradmax*10, '--')
    plt.plot(xl, xl*0-opt.gradmax*10, '--')
    plt.ylabel('grad.ampl. [mT/m]')
    plt.legend(['x', 'y'], loc='upper left')
    plt.title('gradient amplitude')

    plt.subplot(2,3,6)
    plt.plot(model.slew[0,:,:].flatten().detach().cpu()/100, '.-')
    plt.plot(model.slew[1,:,:].flatten().detach().cpu()/100, '.-')
    xl = torch.tensor(plt.xlim())
    plt.plot(xl, xl*0+opt.slewmax/100, '--')
    plt.plot(xl, xl*0-opt.slewmax/100, '--')
    plt.ylabel('slew rate [T/m/s]')
    plt.title('slew rate')

    plt.suptitle(f'epoch {epoch}')

    if do_save:
        plt.savefig(f'out/epoch{epoch:04d}.png') # for animation later
    plt.show()
    
    
def traj2phys(ktraj, res=2e-3):
    # res in m!
    return ktraj / 2 / torch.pi / res

def traj2norm(ktraj, res=2e-3):
    # res in m!
    return ktraj * 2 * torch.pi * res

def eddy_perturbation(ktraj, opt, ampl=1e-5, alphas=None, taus=None):
    """
    simple eddy current (EC) forward model
    inputs:
        ktraj:  k-space locations, physical units! [2, nSamplingPoints]
        opt:    options structure
        ampl:   global scaling of eddy current strength
        alphas: amplitudes of individual EC components
        taus:   time constants of individual EC components
    
    returns:
        k_perturbed: k-space locations after applying multi-exponential EC model to gradient waveforms

    """
    ktraj_phys = ktraj.clone().reshape([2,opt.num_shots,opt.nfe]) # fix shape

    # k-trajectory to gradient waveforms (finite differences / derivative)
    grad = (ktraj_phys[:,:,1:] - ktraj_phys[:,:,:-1]) / opt.dt / (gamma_*1e6) # [T/m]

    # gradient waveforms to slew rate (finite differences / derivative)
    slew = (grad[:,:,1:] - grad[:,:,:-1]) / opt.dt 
    slew = torch.cat([slew, torch.zeros([2,opt.num_shots,1],device=slew.device)], dim=2) # preserve shape of grad

    # time axis 
    timings = torch.arange(0, grad.shape[-1]*opt.dt, opt.dt, device=grad.device)

    # generate eddy current (EC) kernel:
    # simple multi-exponential model here, see Jehenson et al., doi:10.1016/0022-2364(90)90133-T
    # and doi:10.1002/mrm.70093
    if alphas is None:
        alphas = [   1,    0] # amplitudes of EC components
    if taus is None:
        taus   = [50e-6, 1e-1] # time constants of EC components
    ec_perturb = torch.zeros(timings.shape, device=slew.device)
    for alpha, tau in zip(alphas, taus): # Sum up all exponentials
        ec_perturb += alpha*torch.exp(-timings/tau)

    # Use neural network convolution
    response = torch.nn.functional.conv1d(
        slew.reshape([2*opt.num_shots,1,-1]), # [batch,channels=1,time]
        ec_perturb.flip(0).unsqueeze(0).unsqueeze(0), # Flip as conv in machine learning terms is actually cross-correlation, add singleton for batch & channel.
        padding=len(ec_perturb)
        ).reshape(2,opt.num_shots,-1)[:,:,:len(ec_perturb)] # bring back to reasonable shape

    grad_perturbed = grad - ampl * response # Minus due to Lenz's law.

    # cumulative sum (~integration) to get back from gradient waveforms to k-space trajectory
    k_perturbed = torch.cumsum(
            torch.cat([ktraj_phys[:,:,0].unsqueeze(-1), # start integration at original k-value
                    grad_perturbed * opt.dt * (gamma_*1e6)], dim=2),
        dim=2) 
    
    return k_perturbed


def set_misc_params(opt):
    # more misc params (required by SNOPY framework, but not used in this simple demo)
    opt.save_latest_freq = 5000
    opt.save_epoch_freq = 40
    opt.val_epoch_freq = 40
    opt.phase = 'train'
    opt.train_phase = 'generator'
    opt.which_epoch = 'latest'

    opt.dataroot = None
    opt.batchSize = 1
    opt.checkpoints_dir = './checkpoints'

    opt.verbose = True
    opt.suffix = 'simpleSNOPY'

    opt.isTrain = True
    opt.resize_or_crop = False
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.norm = 'instance'
    opt.beta1 = 0.5
    opt.contrast_condition = None
    opt.epoch_count = 1
    opt.continue_train = False
    opt.ReconVSTraj = 1 # scaling factor of trajectory learning rate
    

def pulseq_radial_out(Nx, Nr, ros=2, fov=220e-3, alpha=10, slice_thickness=3e-3, TR=10, do_plot=True):
 
    # adapted from https://github.com/imr-framework/pypulseq/blob/master/examples/scripts/write_radial_gre.py
    seq = pp.Sequence()

    delta = 2*np.pi / Nr  # Angular increment

    rf_spoiling_inc = 117  # RF spoiling increment

    # Set system limits
    system = pp.Opts(
        max_grad=50,
        grad_unit="mT/m",
        max_slew=180,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    # ======
    # CREATE EVENTS
    # ======
    # Create alpha-degree slice selection pulse and gradient
    rf, gz, _ = pp.make_sinc_pulse(
        apodization=0.5,
        duration=4e-3,
        flip_angle=alpha * np.pi / 180,
        slice_thickness=slice_thickness,
        system=system,
        time_bw_product=4,
        return_gz=True,
    )

    # Define other gradients and ADC events
    deltak = 1 / fov
    gx = pp.make_trapezoid(
        channel="x", flat_area=Nx * deltak / 2, flat_time=6.4e-3 / 5, system=system
    )
    gy = pp.make_trapezoid(
        channel="x", flat_area=0, flat_time=6.4e-3 / 5, system=system
    )
    adc = pp.make_adc(
        num_samples=ros*Nx, duration=gx.flat_time, delay=gx.rise_time, system=system
    )

    gz_reph = pp.make_trapezoid(
        channel="z", area=-gz.area / 2, duration=2e-3, system=system
    )
    # Gradient spoiling
    gx_spoil = pp.make_trapezoid(channel="x", area=0.5 * Nx * deltak, system=system)
    gz_spoil = pp.make_trapezoid(channel="z", area=4 / slice_thickness, system=system)

    # Calculate timing
    delay_TR = (
        np.ceil(
            (
                TR
                - pp.calc_duration(gz)
                - pp.calc_duration(gx)
            )
            / seq.grad_raster_time
        )
        * seq.grad_raster_time
    )
    assert np.all(delay_TR) > pp.calc_duration(gx_spoil, gz_spoil)
    rf_phase = 0
    rf_inc = 0

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    for i in range(Nr):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi

        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_inc + rf_phase, 360.0)[1]

        seq.add_block(rf, gz)
        phi = delta * (i - 1)
        seq.add_block(*pp.rotate(gz_reph, angle=phi, axis="z"))

        seq.add_block(*pp.rotate(gx, gy, adc, angle=phi, axis="z"))

        seq.add_block(
            *pp.rotate(gx_spoil, gz_spoil, pp.make_delay(delay_TR), angle=phi, axis="z")
        )

    

    k_traj_adc, k_traj, _, _, t_adc = seq.calculate_kspace()
    knorm = np.reshape(k_traj_adc * fov / Nx * 2 * np.pi, [3, Nr, -1])
    
    if do_plot:
        seq.plot()
        
        plt.figure()
        for shot in range(Nr):
            plt.plot(knorm[0,shot,:], knorm[1,shot,:], '.-')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.title('trajectory')
    
    
    return seq, knorm