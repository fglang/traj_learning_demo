import torch
import matplotlib.pyplot as plt

gamma_ = 42.5764 # MHz/T

def plot_traj(ktraj, num_shots, nfe):
    ''' plot of k-space trajectory '''

    traj_temp = ktraj.reshape([2,num_shots,nfe]).detach().cpu()
    for ishot in range(num_shots):
        plt.plot(traj_temp[0,ishot,:], traj_temp[1,ishot,:], '.-')
    plt.title('trajectory')
    plt.xlabel('kx'), plt.ylabel('ky')


def plot_epoch(model, opt, all_losses, epoch, do_save=True):
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
    plt.imshow(Iunder_mag.cpu().detach(), vmin=0, vmax=Ireal_mag.max())
    plt.colorbar()
    plt.title('adjoint')

    plt.subplot(2,4,3)
    plt.imshow(Ifake_mag.cpu().detach(), vmin=0, vmax=Ireal_mag.max())
    plt.colorbar()
    plt.title('CG SENSE')

    plt.subplot(2,4,4)
    plt.imshow(diff_img.cpu().detach(), vmin=-0.2, vmax=0.2)
    plt.colorbar()
    plt.title('diff: ground truth - CG SENSE')

    ### loss curves
    plt.subplot(2,3,4)
    for loss_name in model.loss_names:
        plt.plot(all_losses[loss_name], '.-', label=loss_name)
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.title('losses')

    ### gradient waveforms and slew rates
    plt.subplot(2,3,5)
    plt.plot(model.grad[0,:,:].flatten().detach().cpu()*10, '.-')
    plt.plot(model.grad[1,:,:].flatten().detach().cpu()*10, '.-')
    plt.ylabel('grad.ampl. [mT/m]')
    plt.legend(['x', 'y'], loc='upper left')
    plt.title('gradient amplitude')

    plt.subplot(2,3,6)
    plt.plot(model.slew[0,:,:].flatten().detach().cpu()/100, '.-')
    plt.plot(model.slew[1,:,:].flatten().detach().cpu()/100, '.-')
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
    """
    ktraj_phys = ktraj.clone().reshape([2,opt.num_shots,opt.nfe])

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
