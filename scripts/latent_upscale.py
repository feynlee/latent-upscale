import cv2
import modules.scripts as scripts
import modules.shared as shared
import gradio as gr
import numpy as np
import torch
import hashlib
import inspect
import k_diffusion.sampling
import modules.shared as shared

from PIL import Image, ImageOps
from modules import images, masking, sd_samplers_common
from modules.processing import setup_color_correction, create_random_tensors
from modules.shared import opts, state
from modules.sd_samplers import find_sampler_config
from modules import sd_samplers
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
        print(f"opts.img2img2img_fix_steps: {opts.img2img2img_fix_steps}")
        opts.img2img2img_fix_steps = True
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas = self.get_sigmas(p, steps)
        sigma_sched = sigmas[steps - t_enc - 1:]

        # ----------------
        # modified code
        # ----------------
        # t_enc = steps
        # new_steps = int(steps/p.denoising_strength)
        # sigmas = self.get_sigmas(p, new_steps)
        # sigma_sched = sigmas[-(steps + 1):]

        xi = x + noise * sigma_sched[0]
        print(f"steps: {steps}, new_steps: {t_enc}")
        print(f"sigmas: {sigmas}")
        print(f"sigma_sched: {sigma_sched}")

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
        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

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


opt_C = 4
opt_f = 8


class Script(scripts.Script):

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "Latent Upscale"


# Determines when the script should be shown in the dropdown menu via the
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):

        return is_img2img
        # return scripts.AlwaysVisible

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        upscale_method = gr.Dropdown(["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"], label="Upscale method")
        scheduler = gr.Dropdown(["simple", "normal", "karras", "exponential", "polyexponential", "automatic"], label="Scheduler")
        return [upscale_method, scheduler]



# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, upscale_method, scheduler):
        p.upscale_method = upscale_method
        print(f"set Upscale method in run: {upscale_method}")
        print(f"scheduler: {scheduler}")
        # print(f"p.scripts is None: {p.scripts is None}")
        # print("always on scripts:")
        # print(p.scripts.alwayson_scripts)
        # p.scripts.alwayson_scripts += p.scripts.selectable_scripts
        # print("always on scripts:")
        # print(p.scripts.alwayson_scripts)

        # add custom schedulers: p.sampler_noise_scheduler_override
        def simple_scheduler(model, steps, device='cuda'):
            sigs = []
            ss = len(model.sigmas) / steps
            for x in range(steps):
                sigs += [float(model.sigmas[-(1 + int(x * ss))])]
            sigs += [0.0]
            return torch.FloatTensor(sigs).to(device)

        def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
            if ddim_discr_method == 'uniform':
                c = num_ddpm_timesteps // num_ddim_timesteps
                ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
            elif ddim_discr_method == 'quad':
                ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
            else:
                raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

            # assert ddim_timesteps.shape[0] == num_ddim_timesteps
            # add one to get the final alpha values right (the ones from first scale to data during sampling)
            steps_out = ddim_timesteps + 1
            if verbose:
                print(f'Selected timesteps for ddim sampler: {steps_out}')
            return steps_out

        def ddim_scheduler(model, steps, device='cuda'):
            sigs = []
            ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=steps, num_ddpm_timesteps=model.inner_model.inner_model.num_timesteps, verbose=False)
            for x in range(len(ddim_timesteps) - 1, -1, -1):
                ts = ddim_timesteps[x]
                if ts > 999:
                    ts = 999
                sigs.append(model.t_to_sigma(torch.tensor(ts)))
            sigs += [0.0]
            return torch.FloatTensor(sigs).to(device)


        def sampler_noise_scheduler_override(steps):
            denoiser = k_diffusion.external.CompVisVDenoiser if p.sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
            model_wrap = denoiser(p.sd_model, quantize=opts.enable_quantization)

            if scheduler == "karras":
                sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item())
                sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device='cuda')
            elif scheduler == "exponential":
                m_sigma_min, m_sigma_max = (model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item())
                sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (m_sigma_min, m_sigma_max)
                sigmas = k_diffusion.sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device='cuda')
            elif scheduler == "polyexponential":
                m_sigma_min, m_sigma_max = (model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item())
                sigma_min, sigma_max = (0.1, 10) if opts.use_old_karras_scheduler_sigmas else (m_sigma_min, m_sigma_max)
                sigmas = k_diffusion.sampling.get_sigmas_polyexponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device='cuda')
            elif scheduler == "normal":
                sigmas = model_wrap.get_sigmas(steps)
            elif scheduler == "simple":
                sigmas = simple_scheduler(model_wrap, steps)
            elif scheduler == "ddim_uniform":
                sigmas = ddim_scheduler(model_wrap, steps)
            else:
                print("error invalid scheduler", scheduler)

            print(f"sigmas device: {sigmas.device}")
            return sigmas

        if scheduler in ["simple", "normal", "karras", "exponential", "polyexponential"]:
            p.sampler_noise_scheduler_override = sampler_noise_scheduler_override

        # override the init method
        def init(all_prompts, all_seeds, all_subseeds, **kwargs):
            # p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
            p.sampler = KDiffusionSampler(samplers_k_diffusion_dict[p.sampler_name], p.sd_model)
            p.sampler.config = find_sampler_config(p.sampler_name)

            crop_region = None

            image_mask = p.image_mask

            if image_mask is not None:
                image_mask = image_mask.convert('L')

                if p.inpainting_mask_invert:
                    image_mask = ImageOps.invert(image_mask)

                if p.mask_blur_x > 0:
                    np_mask = np.array(image_mask)
                    kernel_size = 2 * int(4 * p.mask_blur_x + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
                    image_mask = Image.fromarray(np_mask)

                if p.mask_blur_y > 0:
                    np_mask = np.array(image_mask)
                    kernel_size = 2 * int(4 * p.mask_blur_y + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
                    image_mask = Image.fromarray(np_mask)

                if p.inpaint_full_res:
                    p.mask_for_overlay = image_mask
                    mask = image_mask.convert('L')
                    crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
                    crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)
                    x1, y1, x2, y2 = crop_region

                    mask = mask.crop(crop_region)
                    image_mask = images.resize_image(2, mask, p.width, p.height)
                    p.paste_to = (x1, y1, x2-x1, y2-y1)
                else:
                    image_mask = images.resize_image(p.resize_mode, image_mask, p.width, p.height)
                    np_mask = np.array(image_mask)
                    np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                    p.mask_for_overlay = Image.fromarray(np_mask)

                p.overlay_images = []

            latent_mask = p.latent_mask if p.latent_mask is not None else image_mask

            add_color_corrections = opts.img2img_color_correction and p.color_corrections is None
            if add_color_corrections:
                p.color_corrections = []
            imgs = []
            for img in p.init_images:

                # Save init image
                if opts.save_init_img:
                    p.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=p.init_img_hash, save_to_dirs=False)

                image = images.flatten(img, opts.img2img_background_color)

                if crop_region is None and p.resize_mode != 3:
                    image = images.resize_image(p.resize_mode, image, p.width, p.height)

                if image_mask is not None:
                    image_masked = Image.new('RGBa', (image.width, image.height))
                    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(p.mask_for_overlay.convert('L')))

                    p.overlay_images.append(image_masked.convert('RGBA'))

                # crop_region is not None if we are doing inpaint full res
                if crop_region is not None:
                    image = image.crop(crop_region)
                    image = images.resize_image(2, image, p.width, p.height)

                if image_mask is not None:
                    if p.inpainting_fill != 1:
                        image = masking.fill(image, latent_mask)

                if add_color_corrections:
                    p.color_corrections.append(setup_color_correction(image))

                image = np.array(image).astype(np.float32) / 255.0
                image = np.moveaxis(image, 2, 0)

                imgs.append(image)

            if len(imgs) == 1:
                batch_images = np.expand_dims(imgs[0], axis=0).repeat(p.batch_size, axis=0)
                if p.overlay_images is not None:
                    p.overlay_images = p.overlay_images * p.batch_size

                if p.color_corrections is not None and len(p.color_corrections) == 1:
                    p.color_corrections = p.color_corrections * p.batch_size

            elif len(imgs) <= p.batch_size:
                p.batch_size = len(imgs)
                batch_images = np.array(imgs)
            else:
                raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {p.batch_size} or less")

            image = torch.from_numpy(batch_images)
            image = 2. * image - 1.
            image = image.to(shared.device)

            p.init_latent = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image))

            if p.resize_mode == 3:
                # ----------------
                # modified code
                # ----------------
                p.init_latent = torch.nn.functional.interpolate(p.init_latent, size=(p.height // opt_f, p.width // opt_f), mode=p.upscale_method)

            if image_mask is not None:
                init_mask = latent_mask
                latmask = init_mask.convert('RGB').resize((p.init_latent.shape[3], p.init_latent.shape[2]))
                latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
                latmask = latmask[0]
                latmask = np.around(latmask)
                latmask = np.tile(latmask[None], (4, 1, 1))

                p.mask = torch.asarray(1.0 - latmask).to(shared.device).type(p.sd_model.dtype)
                p.nmask = torch.asarray(latmask).to(shared.device).type(p.sd_model.dtype)

                # this needs to be fixed to be done in sample() using actual seeds for batches
                if p.inpainting_fill == 2:
                    p.init_latent = p.init_latent * p.mask + create_random_tensors(p.init_latent.shape[1:], all_seeds[0:p.init_latent.shape[0]]) * p.nmask
                elif p.inpainting_fill == 3:
                    p.init_latent = p.init_latent * p.mask

            p.image_conditioning = p.img2img_image_conditioning(image, p.init_latent, image_mask)
            print("new init executed")

        p.init = init
