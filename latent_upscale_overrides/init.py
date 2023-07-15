import cv2
import modules.shared as shared
import numpy as np
import torch
import hashlib

from PIL import Image, ImageOps
from modules import images, masking
from modules.processing import setup_color_correction, create_random_tensors
from modules.shared import opts
from modules import sd_samplers


opt_C = 4
opt_f = 8


def parse_upscale_method(upscale_method):
    if upscale_method in ("bilinear-antialiased", "bicubic-antialiased"):
        upscale_method = upscale_method.split('-')[0]
        antialias = True
    else:
        antialias = False
    return upscale_method, antialias


def init(p, upscale_method, all_prompts, all_seeds, all_subseeds, **kwargs):
    # ------------------------------------------------
    # modified code: use the correct number of steps
    # ------------------------------------------------
    opts.img2img_fix_steps = True

    p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)

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
            crop_region = masking.get_crop_region(np.array(mask),
                                                    p.inpaint_full_res_padding)
            crop_region = masking.expand_crop_region(
                crop_region, p.width, p.height, mask.width, mask.height)
            x1, y1, x2, y2 = crop_region

            mask = mask.crop(crop_region)
            image_mask = images.resize_image(2, mask, p.width, p.height)
            p.paste_to = (x1, y1, x2-x1, y2-y1)
        else:
            image_mask = images.resize_image(p.resize_mode, image_mask,
                                                p.width, p.height)
            np_mask = np.array(image_mask)
            np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255
                                ).astype(np.uint8)
            p.mask_for_overlay = Image.fromarray(np_mask)

        p.overlay_images = []

    latent_mask = p.latent_mask if p.latent_mask is not None else image_mask

    add_color_corrections = opts.img2img_color_correction \
                            and p.color_corrections is None
    if add_color_corrections:
        p.color_corrections = []
    imgs = []
    for img in p.init_images:

        # Save init image
        if opts.save_init_img:
            p.init_img_hash = hashlib.md5(img.tobytes()).hexdigest()
            images.save_image(img, path=opts.outdir_init_images,
                                basename=None, forced_filename=p.init_img_hash,
                                save_to_dirs=False)

        image = images.flatten(img, opts.img2img_background_color)

        if crop_region is None and p.resize_mode != 3:
            image = images.resize_image(p.resize_mode, image, p.width, p.height)

        if image_mask is not None:
            image_masked = Image.new('RGBa', (image.width, image.height))
            image_masked.paste(
                image.convert("RGBA").convert("RGBa"),
                mask=ImageOps.invert(p.mask_for_overlay.convert('L')))

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
        batch_images = np.expand_dims(imgs[0], axis=0).repeat(p.batch_size,
                                                                axis=0)
        if p.overlay_images is not None:
            p.overlay_images = p.overlay_images * p.batch_size

        if p.color_corrections is not None and len(p.color_corrections) == 1:
            p.color_corrections = p.color_corrections * p.batch_size

    elif len(imgs) <= p.batch_size:
        p.batch_size = len(imgs)
        batch_images = np.array(imgs)
    else:
        raise RuntimeError(
            f"bad number of images passed: {len(imgs)}; "
            f"expecting {p.batch_size} or less")

    image = torch.from_numpy(batch_images)
    image = 2. * image - 1.
    image = image.to(shared.device)

    p.init_latent = p.sd_model.get_first_stage_encoding(
        p.sd_model.encode_first_stage(image))

    if p.resize_mode == 3:
        # -------------------------------------------
        # modified code: pass in the upscale method
        # -------------------------------------------
        upscale_method, antialias = parse_upscale_method(upscale_method)
        p.init_latent = torch.nn.functional.interpolate(
            p.init_latent,
            size=(p.height // opt_f, p.width // opt_f),
            mode=upscale_method,
            antialias=antialias)

    if image_mask is not None:
        init_mask = latent_mask
        latmask = init_mask.convert('RGB').resize(
            (p.init_latent.shape[3], p.init_latent.shape[2]))
        latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
        latmask = latmask[0]
        latmask = np.around(latmask)
        latmask = np.tile(latmask[None], (4, 1, 1))

        p.mask = torch.asarray(1.0 - latmask).to(shared.device).type(
            p.sd_model.dtype)
        p.nmask = torch.asarray(latmask).to(shared.device).type(
            p.sd_model.dtype)

        # this needs to be fixed to be done in sample()
        # using actual seeds for batches
        if p.inpainting_fill == 2:
            p.init_latent = p.init_latent * p.mask + create_random_tensors(
                p.init_latent.shape[1:],
                all_seeds[0:p.init_latent.shape[0]]) * p.nmask
        elif p.inpainting_fill == 3:
            p.init_latent = p.init_latent * p.mask

    p.image_conditioning = p.img2img_image_conditioning(
        image, p.init_latent, image_mask)
    print("new init executed")