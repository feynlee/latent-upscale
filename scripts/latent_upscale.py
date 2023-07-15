import modules.scripts as scripts
import gradio as gr

from latent_upscale_overrides.schedulers import sampler_noise_scheduler_override
from latent_upscale_overrides.init import init


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
        with gr.Row():
            upscale_method = gr.Dropdown(["bilinear", "bilinear-antialiased",
                                          "bicubic", "bicubic-antialiased",
                                          "linear", "trilinear",
                                          "area", "nearest",  "nearest-exact"],
                                        label="Upscale method")
            scheduler = gr.Dropdown(["simple", "normal", "karras", "exponential",
                                    "polyexponential", "automatic"],
                                    label="Scheduler")
        return [upscale_method, scheduler]



# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, upscale_method, scheduler):
        print(f"set Upscale method in run: {upscale_method}")
        print(f"scheduler: {scheduler}")

        # override the sampler_noise_scheduler_override method
        # if the selected method is not recognized, use the default scheduler
        if scheduler in ["simple", "normal", "karras",
                         "exponential", "polyexponential", "ddim_uniform"]:
            p.sampler_noise_scheduler_override \
                = lambda steps: sampler_noise_scheduler_override(p, scheduler, steps)

        # override the init method
        p.init = lambda all_prompts, all_seeds, all_subseeds, **kwargs: init(
            p, upscale_method, all_prompts, all_seeds, all_subseeds, **kwargs)
