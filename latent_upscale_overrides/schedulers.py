import torch
import k_diffusion.sampling
import numpy as np

from modules.shared import opts


def simple_scheduler(model, steps, device='cuda'):
    sigs = []
    ss = len(model.sigmas) / steps
    for x in range(steps):
        sigs += [float(model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps,
                        num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8),
                                        num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right
    # (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

def ddim_scheduler(model, steps, device='cuda'):
    sigs = []
    ddim_timesteps = make_ddim_timesteps(
        ddim_discr_method="uniform",
        num_ddim_timesteps=steps,
        num_ddpm_timesteps=model.inner_model.inner_model.num_timesteps,
        verbose=False)
    for x in range(len(ddim_timesteps) - 1, -1, -1):
        ts = ddim_timesteps[x]
        if ts > 999:
            ts = 999
        sigs.append(model.t_to_sigma(torch.tensor(ts)))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


# use custom schedulers: p.sampler_noise_scheduler_override
def sampler_noise_scheduler_override(p, scheduler, steps):
    if p.sd_model.parameterization == "v":
        denoiser = k_diffusion.external.CompVisVDenoiser
    else:
        denoiser = k_diffusion.external.CompVisDenoiser
    model_wrap = denoiser(p.sd_model, quantize=opts.enable_quantization)

    if scheduler == "karras":
        if opts.use_old_karras_scheduler_sigmas:
            sigma_min, sigma_max = (0.1, 10)
        else:
            sigma_min, sigma_max = (model_wrap.sigmas[0].item(),
                                    model_wrap.sigmas[-1].item())
        sigmas = k_diffusion.sampling.get_sigmas_karras(
            n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device='cuda')
    elif scheduler == "exponential":
        m_sigma_min, m_sigma_max = (model_wrap.sigmas[0].item(),
                                    model_wrap.sigmas[-1].item())
        if opts.use_old_karras_scheduler_sigmas:
            sigma_min, sigma_max = (0.1, 10)
        else:
            sigma_min, sigma_max = (m_sigma_min, m_sigma_max)
        sigmas = k_diffusion.sampling.get_sigmas_exponential(
            n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device='cuda')
    elif scheduler == "polyexponential":
        m_sigma_min, m_sigma_max = (model_wrap.sigmas[0].item(),
                                    model_wrap.sigmas[-1].item())
        if opts.use_old_karras_scheduler_sigmas:
            sigma_min, sigma_max = (0.1, 10)
        else:
            sigma_min, sigma_max = (m_sigma_min, m_sigma_max)
        sigmas = k_diffusion.sampling.get_sigmas_polyexponential(
            n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device='cuda')
    elif scheduler == "normal":
        sigmas = model_wrap.get_sigmas(steps)
    elif scheduler == "simple":
        sigmas = simple_scheduler(model_wrap, steps)
    elif scheduler == "ddim_uniform":
        sigmas = ddim_scheduler(model_wrap, steps)
    else:
        print("error invalid scheduler", scheduler)

    print(f"sigmas device: {sigmas.device}")
    print(f"sigmas: {sigmas}")
    return sigmas