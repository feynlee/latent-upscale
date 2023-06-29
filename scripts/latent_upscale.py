import modules.scripts as scripts
import gradio as gr
import torch
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
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
        upscale_method = gr.Dropdown(["nearest", "bicubic", "area", "lanczos", "bilinear"], label="Upscale method")
        return [upscale_method]



# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, upscale_method):

        p.init_latent = torch.nn.functional.interpolate(p.init_latent, size=(p.height // opt_f, p.width // opt_f), mode=upscale_method)
        proc = process_images(p)

        return proc