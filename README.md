# Latent Upscale

<a href="https://ko-fi.com/ziyueli"><img src="https://storage.ko-fi.com/cdn/brandasset/kofi_button_blue.png?_gl=1*1lrplqr*_ga*MjA3NzkyMTU3Mi4xNjgxMDg2MjQw*_ga_M13FZ7VQ2C*MTY4MTA4NjI0Ni4xLjEuMTY4MTA4NzYxNi40NC4wLjA." alt="Ko-fi" width="180px"></a>

Enhance the latent upscale options in the img2img process in [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to provide more flexibility and better image quality.

## Motivation

1. The current img2img process lacks the ability to select different options for Latent Upscale.
2. The default latent upscale method ("bilinear") often produces blurry images.

This plugin introduces alternative interpolation methods for upscaling and offers different schedulers for the diffusion process, resulting in superior upscaled images.
Moreover, this plugin expands the upscale options available in the Latent Space, surpassing those offered by the "Hires Fix" for the txt2img process.

## Example Comparison

Original Image:

![](assets/original.png)

The default latent upscale (choose Resize mode "latent upscale"):
![](assets/default.png)

Latent Upscale Plugin (Upscale method: "nearest-exact", Scheduler: "simple")
![](assets/nearest-exact-simple8.png)

## Installation

You can find the "Latent Upscale" plugin in the **Available** section under the **Extensions** tab in the WebUI.
Simply search for "Latent Upscale" in the extension search bar to locate it and click on "install".

Alternatively, you can install this plugin by navigating to **Install from URL** under the **Extensions** tab, and then pasting this repo's URL and clicking on **Install**.

![](assets/installation.png)

_Don't forget to go back to **Installed** and click on **Apply**._

## Usage

![](assets/usage.png)

At the bottom of the **img2img** tab, you can select the newly installed **Latent Upscale** script in the **Script** dropdown menu.

To benefit from these enhancements, make sure you have the "Just resize (latent upscale)" option selected for **Resize mode**.
Additionally, all the parameters present in the user interface remain applicable, alongside the new options provided by this plugin in "Upscale Method" and "Scheduler".

## Tips

1. Opting for "nearest", "nearest-exact", or "area" in the Upscale Method and "simple" or "normal" in the Scheduler can often lead to crisper images.
   However, don't hesitate to experiment with different choices to discover the best outcome.
2. If you want more details in the image, you can try to increase the number of steps in the diffusion process.
   However, this will also increase the time required to generate the image.

## Technical Details

1. This plugin overrides the default `init` method for `StableDiffusionProcessingImg2Img` to include additional features:

   1. It adds the option to choose the "Upscale Method" interpolation method when creating the latent image.
   2. It ensures that the diffusion process runs for the correct number of steps, as specified by the user, by setting opts.img2img_fix_steps = True. It is unclear why this was not the default setting for img2img.

2. This plugin assigns the `sampler_noise_scheduler_override` method for `StableDiffusionProcessingImg2Img` so that our custom schedulers can be used for the diffusion process.
