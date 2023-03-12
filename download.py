# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import os
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation


def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    # Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    CNET_MODEL = os.getenv("CNET_MODEL")
    SD_MODEL = os.getenv("SD_MODEL")

    controlnet = ControlNetModel.from_pretrained(
        CNET_MODEL, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL, controlnet=controlnet, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

if __name__ == "__main__":
    download_model()
