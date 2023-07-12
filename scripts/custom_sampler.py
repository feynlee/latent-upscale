from collections import deque
import torch
import inspect
import k_diffusion.sampling
from modules import prompt_parser, devices, sd_samplers_common

from modules.shared import opts, state
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback
from modules.sd_samplers_kdiffusion import CFGDenoiser, TorchHijack, sampler_extra_params, k_diffusion_scheduler


samplers_k_diffusion_dict = {
    'Euler a': 'sample_euler_ancestral',
    'Euler': 'sample_euler',
    'LMS': 'sample_lms',
    'Heun': 'sample_heun',
    'DPM2': 'sample_dpm_2',
    'DPM2 a': 'sample_dpm_2_ancestral',
    'DPM++ 2S a': 'sample_dpmpp_2s_ancestral',
    'DPM++ 2M': 'sample_dpmpp_2m',
    'DPM++ SDE': 'sample_dpmpp_sde',
    'DPM++ 2M SDE': 'sample_dpmpp_2m_sde',
    'DPM fast': 'sample_dpm_fast',
    'DPM adaptive': 'sample_dpm_adaptive',
    'LMS Karras': 'sample_lms',
    'DPM2 Karras': 'sample_dpm_2',
    'DPM2 a Karras': 'sample_dpm_2_ancestral',
    'DPM++ 2S a Karras': 'sample_dpmpp_2s_ancestral',
    'DPM++ 2M Karras': 'sample_dpmpp_2m',
    'DPM++ SDE Karras': 'sample_dpmpp_sde',
    'DPM++ 2M SDE Karras': 'sample_dpmpp_2m_sde',
}


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        denoiser = k_diffusion.external.CompVisVDenoiser if sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser

        self.model_wrap = denoiser(sd_model, quantize=shared.opts.enable_quantization)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.extra_params = sampler_extra_params.get(funcname, [])
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None

        self.conditioning_key = sd_model.model.conditioning_key

    def callback_state(self, d):
        step = d['i']
        latent = d["denoised"]
        if opts.live_preview_content == "Combined":
            sd_samplers_common.store_latent(latent)
        self.last_latent = latent

        if self.stop_at is not None and step > self.stop_at:
            raise sd_samplers_common.InterruptedException

        state.sampling_step = step
        shared.total_tqdm.update()

    def launch_sampling(self, steps, func):
        state.sampling_steps = steps
        state.sampling_step = 0

        try:
            return func()
        except RecursionError:
            print(
                'Encountered RecursionError during sampling, returning last latent. '
                'rho >5 with a polyexponential scheduler may cause this error. '
                'You should try to use a smaller rho value instead.'
            )
            return self.last_latent
        except sd_samplers_common.InterruptedException:
            return self.last_latent

    def number_of_needed_noises(self, p):
        return p.steps

    def initialize(self, p):
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.eta = p.eta if p.eta is not None else opts.eta_ancestral
        self.s_min_uncond = getattr(p, 's_min_uncond', 0.0)

        k_diffusion.sampling.torch = TorchHijack(self.sampler_noises if self.sampler_noises is not None else [])

        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        if 'eta' in inspect.signature(self.func).parameters:
            if self.eta != 1.0:
                p.extra_generation_params["Eta"] = self.eta

            extra_params_kwargs['eta'] = self.eta

        return extra_params_kwargs

    def get_sigmas(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif opts.k_sched_type != "Automatic":
            m_sigma_min, m_sigma_max = (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())
            sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (m_sigma_min, m_sigma_max)
            sigmas_kwargs = {
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
            }

            sigmas_func = k_diffusion_scheduler[opts.k_sched_type]
            p.extra_generation_params["Schedule type"] = opts.k_sched_type

            if opts.sigma_min != m_sigma_min and opts.sigma_min != 0:
                sigmas_kwargs['sigma_min'] = opts.sigma_min
                p.extra_generation_params["Schedule min sigma"] = opts.sigma_min
            if opts.sigma_max != m_sigma_max and opts.sigma_max != 0:
                sigmas_kwargs['sigma_max'] = opts.sigma_max
                p.extra_generation_params["Schedule max sigma"] = opts.sigma_max

            default_rho = 1. if opts.k_sched_type == "polyexponential" else 7.

            if opts.k_sched_type != 'exponential' and opts.rho != 0 and opts.rho != default_rho:
                sigmas_kwargs['rho'] = opts.rho
                p.extra_generation_params["Schedule rho"] = opts.rho

            sigmas = sigmas_func(n=steps, **sigmas_kwargs, device=shared.device)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (self.model_wrap.sigmas[0].item(), self.model_wrap.sigmas[-1].item())

            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas

    def create_noise_sampler(self, x, sigmas, p):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        if shared.opts.no_dpmpp_sde_batch_determinism:
            return None

        from k_diffusion.sampling import BrownianTreeNoiseSampler
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        sigmas = self.get_sigmas(p, steps)

        sigma_sched = sigmas[steps - t_enc - 1:]
        # xi = x + noise * sigma_sched[0]
        xi = x + noise

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if 'sigma_min' in parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs['sigma_min'] = sigma_sched[-2]
        if 'sigma_max' in parameters:
            extra_params_kwargs['sigma_max'] = sigma_sched[0]
        if 'n' in parameters:
            extra_params_kwargs['n'] = len(sigma_sched) - 1
        if 'sigma_sched' in parameters:
            extra_params_kwargs['sigma_sched'] = sigma_sched
        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigma_sched

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }

        print("use custom KSampler")
        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, sigmas, extra_args=extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps = steps or p.steps

        sigmas = self.get_sigmas(p, steps)

        x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()
            if 'n' in parameters:
                extra_params_kwargs['n'] = steps
        else:
            extra_params_kwargs['sigmas'] = sigmas

        if self.config.options.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler

        self.last_latent = x
        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args={
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }, disable=False, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples

