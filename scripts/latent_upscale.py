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
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)

        sigmas = self.get_sigmas(p, steps)

        sigma_sched = sigmas[steps - t_enc - 1:]
        xi = x + noise * sigma_sched[0]
        print(f"steps: {steps}, t_enc: {t_enc}")
        print(f"sigma_sched: {sigma_sched}")
        print(f"sigmas: {sigmas}")
        # xi = x + noise * sigmas[0]

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
        # if 'sigmas' in parameters:
        #     extra_params_kwargs['sigmas'] = sigma_sched

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
        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, sigma_sched, extra_args=extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

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
                p.init_latent = torch.nn.functional.interpolate(p.init_latent, size=(p.height // opt_f, p.width // opt_f), mode=p.upscale_method)
                # p.init_latent = p.init_latent.clamp(min=0, max=1)

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

#     def process_batch(p, *args, **kwargs):
#         print("Entered script.process_batch")
#         # Do the same procedures as defined in StableDiffusionProcessingImg2Img's __init__
#         # to get the init_latent
#         p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
#         image_mask = p.image_mask
#         crop_region = None

#         if image_mask is not None:
#             image_mask = image_mask.convert('L')

#             if p.inpainting_mask_invert:
#                 image_mask = ImageOps.invert(image_mask)

#             if p.mask_blur_x > 0:
#                 np_mask = np.array(image_mask)
#                 kernel_size = 2 * int(4 * p.mask_blur_x + 0.5) + 1
#                 np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), p.mask_blur_x)
#                 image_mask = Image.fromarray(np_mask)

#             if p.mask_blur_y > 0:
#                 np_mask = np.array(image_mask)
#                 kernel_size = 2 * int(4 * p.mask_blur_y + 0.5) + 1
#                 np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), p.mask_blur_y)
#                 image_mask = Image.fromarray(np_mask)

#             if p.inpaint_full_res:
#                 p.mask_for_overlay = image_mask
#                 mask = image_mask.convert('L')
#                 crop_region = masking.get_crop_region(np.array(mask), p.inpaint_full_res_padding)
#                 crop_region = masking.expand_crop_region(crop_region, p.width, p.height, mask.width, mask.height)
#                 x1, y1, x2, y2 = crop_region

#                 mask = mask.crop(crop_region)
#                 image_mask = images.resize_image(2, mask, p.width, p.height)
#                 p.paste_to = (x1, y1, x2-x1, y2-y1)
#             else:
#                 image_mask = images.resize_image(p.resize_mode, image_mask, p.width, p.height)
#                 np_mask = np.array(image_mask)
#                 np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
#                 p.mask_for_overlay = Image.fromarray(np_mask)

#             p.overlay_images = []

#         latent_mask = p.latent_mask if p.latent_mask is not None else image_mask

#         add_color_corrections = opts.img2img_color_correction and p.color_corrections is None
#         if add_color_corrections:
#             p.color_corrections = []
#         imgs = []
#         for img in p.init_images:

#             # Save init image
#             if opts.save_init_img:
#                 p.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
#                 images.save_image(img, path=opts.outdir_init_images, basename=None, forced_filename=p.init_img_hash, save_to_dirs=False)

#             image = images.flatten(img, opts.img2img_background_color)

#             if crop_region is None and p.resize_mode != 3:
#                 image = images.resize_image(p.resize_mode, image, p.width, p.height)

#             if image_mask is not None:
#                 image_masked = Image.new('RGBa', (image.width, image.height))
#                 image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(p.mask_for_overlay.convert('L')))

#                 p.overlay_images.append(image_masked.convert('RGBA'))

#             # crop_region is not None if we are doing inpaint full res
#             if crop_region is not None:
#                 image = image.crop(crop_region)
#                 image = images.resize_image(2, image, p.width, p.height)

#             if image_mask is not None:
#                 if p.inpainting_fill != 1:
#                     image = masking.fill(image, latent_mask)

#             if add_color_corrections:
#                 p.color_corrections.append(setup_color_correction(image))

#             image = np.array(image).astype(np.float32) / 255.0
#             image = np.moveaxis(image, 2, 0)

#             imgs.append(image)

#         if len(imgs) == 1:
#             batch_images = np.expand_dims(imgs[0], axis=0).repeat(p.batch_size, axis=0)
#             if p.overlay_images is not None:
#                 p.overlay_images = p.overlay_images * p.batch_size

#             if p.color_corrections is not None and len(p.color_corrections) == 1:
#                 p.color_corrections = p.color_corrections * p.batch_size

#         elif len(imgs) <= p.batch_size:
#             p.batch_size = len(imgs)
#             batch_images = np.array(imgs)
#         else:
#             raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {p.batch_size} or less")

#         image = torch.from_numpy(batch_images)
#         image = 2. * image - 1.
#         image = image.to(shared.device)

#         p.init_latent = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image))


#         # Upscale the latent space to the desired resolution with options to choose the method
#         p.init_latent = torch.nn.functional.interpolate(p.init_latent, size=(p.height // opt_f, p.width // opt_f), mode=p.upscale_method)

#         p.image_conditioning = p.img2img_image_conditioning(image, p.init_latent, image_mask)

#         print(f"latent upscale with {p.upscale_method} done")


# def process_images(p: StableDiffusionProcessing) -> Processed:
#     if p.scripts is not None:
#         p.scripts.before_process(p)

#     stored_opts = {k: opts.data[k] for k in p.override_settings.keys()}

#     try:
#         # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
#         if sd_models.checkpoint_alisases.get(p.override_settings.get('sd_model_checkpoint')) is None:
#             p.override_settings.pop('sd_model_checkpoint', None)
#             sd_models.reload_model_weights()

#         for k, v in p.override_settings.items():
#             setattr(opts, k, v)

#             if k == 'sd_model_checkpoint':
#                 sd_models.reload_model_weights()

#             if k == 'sd_vae':
#                 sd_vae.reload_vae_weights()

#         sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())

#         res = process_images_inner(p)

#     finally:
#         sd_models.apply_token_merging(p.sd_model, 0)

#         # restore opts to original state
#         if p.override_settings_restore_afterwards:
#             for k, v in stored_opts.items():
#                 setattr(opts, k, v)

#                 if k == 'sd_vae':
#                     sd_vae.reload_vae_weights()

#     return res


# def process_images_inner(p: StableDiffusionProcessing) -> Processed:
#     """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

#     if type(p.prompt) == list:
#         assert(len(p.prompt) > 0)
#     else:
#         assert p.prompt is not None

#     devices.torch_gc()

#     seed = get_fixed_seed(p.seed)
#     subseed = get_fixed_seed(p.subseed)

#     modules.sd_hijack.model_hijack.apply_circular(p.tiling)
#     modules.sd_hijack.model_hijack.clear_comments()

#     comments = {}

#     p.setup_prompts()

#     if type(seed) == list:
#         p.all_seeds = seed
#     else:
#         p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

#     if type(subseed) == list:
#         p.all_subseeds = subseed
#     else:
#         p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

#     def infotext(iteration=0, position_in_batch=0):
#         return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

#     if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
#         model_hijack.embedding_db.load_textual_inversion_embeddings()

#     if p.scripts is not None:
#         p.scripts.process(p)

#     infotexts = []
#     output_images = []

#     with torch.no_grad(), p.sd_model.ema_scope():
#         with devices.autocast():
#             p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

#             # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
#             if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
#                 sd_vae_approx.model()

#             sd_unet.apply_unet()

#         if state.job_count == -1:
#             state.job_count = p.n_iter

#         for n in range(p.n_iter):
#             p.iteration = n

#             if state.skipped:
#                 state.skipped = False

#             if state.interrupted:
#                 break

#             p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
#             p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
#             p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
#             p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

#             if p.scripts is not None:
#                 p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

#             if len(p.prompts) == 0:
#                 break

#             p.parse_extra_network_prompts()

#             if not p.disable_extra_networks:
#                 with devices.autocast():
#                     extra_networks.activate(p, p.extra_network_data)

#             if p.scripts is not None:
#                 p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

#             # params.txt should be saved after scripts.process_batch, since the
#             # infotext could be modified by that callback
#             # Example: a wildcard processed by process_batch sets an extra model
#             # strength, which is saved as "Model Strength: 1.0" in the infotext
#             if n == 0:
#                 with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
#                     processed = Processed(p, [], p.seed, "")
#                     file.write(processed.infotext(p, 0))

#             p.setup_conds()

#             if len(model_hijack.comments) > 0:
#                 for comment in model_hijack.comments:
#                     comments[comment] = 1

#             if p.n_iter > 1:
#                 shared.state.job = f"Batch {n+1} out of {p.n_iter}"

#             with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
#                 samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

#             x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
#             for x in x_samples_ddim:
#                 devices.test_for_nans(x, "vae")

#             x_samples_ddim = torch.stack(x_samples_ddim).float()
#             x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

#             del samples_ddim

#             if lowvram.is_enabled(shared.sd_model):
#                 lowvram.send_everything_to_cpu()

#             devices.torch_gc()

#             if p.scripts is not None:
#                 p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

#             for i, x_sample in enumerate(x_samples_ddim):
#                 p.batch_index = i

#                 x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
#                 x_sample = x_sample.astype(np.uint8)

#                 if p.restore_faces:
#                     if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
#                         images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

#                     devices.torch_gc()

#                     x_sample = modules.face_restoration.restore_faces(x_sample)
#                     devices.torch_gc()

#                 image = Image.fromarray(x_sample)

#                 if p.scripts is not None:
#                     pp = scripts.PostprocessImageArgs(image)
#                     p.scripts.postprocess_image(p, pp)
#                     image = pp.image

#                 if p.color_corrections is not None and i < len(p.color_corrections):
#                     if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
#                         image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
#                         images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
#                     image = apply_color_correction(p.color_corrections[i], image)

#                 image = apply_overlay(image, p.paste_to, i, p.overlay_images)

#                 if opts.samples_save and not p.do_not_save_samples:
#                     images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

#                 text = infotext(n, i)
#                 infotexts.append(text)
#                 if opts.enable_pnginfo:
#                     image.info["parameters"] = text
#                 output_images.append(image)

#                 if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
#                     image_mask = p.mask_for_overlay.convert('RGB')
#                     image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

#                     if opts.save_mask:
#                         images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

#                     if opts.save_mask_composite:
#                         images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

#                     if opts.return_mask:
#                         output_images.append(image_mask)

#                     if opts.return_mask_composite:
#                         output_images.append(image_mask_composite)

#             del x_samples_ddim

#             devices.torch_gc()

#             state.nextjob()

#         p.color_corrections = None

#         index_of_first_image = 0
#         unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
#         if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
#             grid = images.image_grid(output_images, p.batch_size)

#             if opts.return_grid:
#                 text = infotext()
#                 infotexts.insert(0, text)
#                 if opts.enable_pnginfo:
#                     grid.info["parameters"] = text
#                 output_images.insert(0, grid)
#                 index_of_first_image = 1

#             if opts.grid_save:
#                 images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

#     if not p.disable_extra_networks and p.extra_network_data:
#         extra_networks.deactivate(p, p.extra_network_data)

#     devices.torch_gc()

#     res = Processed(
#         p,
#         images_list=output_images,
#         seed=p.all_seeds[0],
#         info=infotext(),
#         comments="".join(f"{comment}\n" for comment in comments),
#         subseed=p.all_subseeds[0],
#         index_of_first_image=index_of_first_image,
#         infotexts=infotexts,
#     )

#     if p.scripts is not None:
#         p.scripts.postprocess(p, res)

#     return res