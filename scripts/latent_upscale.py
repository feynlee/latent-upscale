import cv2
import modules.scripts as scripts
import modules.shared as shared
import gradio as gr
import numpy as np
import torch
import os
import hashlib

from PIL import Image, ImageOps
from modules import images, masking, sd_samplers
from modules.processing import process_images, Processed, setup_color_correction, apply_color_correction
from modules.shared import opts, cmd_opts, state


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

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        upscale_method = gr.Dropdown(["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"], label="Upscale method")
        return [upscale_method]



# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, upscale_method):
        p.upscale_method = upscale_method
        print(f"set Upscale method in run: {upscale_method}")

    def process_batch(p, **kwargs):
        print("Entered script.process_batch")
        # Do the same procedures as defined in StableDiffusionProcessingImg2Img's __init__
        # to get the init_latent
        p.sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
        image_mask = p.image_mask
        crop_region = None

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


        # Upscale the latent space to the desired resolution with options to choose the method
        p.init_latent = torch.nn.functional.interpolate(p.init_latent, size=(p.height // opt_f, p.width // opt_f), mode=p.upscale_method)

        p.image_conditioning = p.img2img_image_conditioning(image, p.init_latent, image_mask)

        print(f"latent upscale with {p.upscale_method} done")