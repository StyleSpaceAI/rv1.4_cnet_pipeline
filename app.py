from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from torch import autocast
import base64
from io import BytesIO
import os
from utils import prepare_image
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_utils import ade_palette

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global midas
    global image_processor
    global image_segmentor

    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    CNET_MODEL = os.getenv("CNET_MODEL")
    SD_MODEL = os.getenv("SD_MODEL")

    controlnet = ControlNetModel.from_pretrained(
        CNET_MODEL, torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL, controlnet=controlnet, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN)
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)

    # model.enable_model_cpu_offload()
    # model.enable_xformers_memory_efficient_attention()

    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    image = model_inputs.get('image', None)
    if image == None:
        return {'message': "No image provided"}
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    image_resolution = model_inputs.get('image_resolution', 512)
    if image_resolution not in [256, 512, 768]:
        return {'message': f"Invalid image_resolution: {image_resolution}"}
    num_inference_steps = model_inputs.get('num_inference_steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9.0)
    if guidance_scale < 0.1 or guidance_scale > 30.0:
        return {'message': f"Invalid scale: {guidance_scale}"}
    seed = model_inputs.get('seed', None)
    eta = model_inputs.get('eta', 0.0)
    negative_prompt = model_inputs.get('negative_prompt', 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
    controlnet_conditioning_scale = model_inputs.get('controlnet_conditioning_scale', 1.0)
    if controlnet_conditioning_scale < 0.0 or controlnet_conditioning_scale > 1.0:
        return {'message': f"Invalid controlnet_conditioning_scale: {controlnet_conditioning_scale}"}
    should_prepare_image = model_inputs.get('should_prepare_image', True)

    image = Image.open(BytesIO(base64.decodebytes(bytes(image, "utf-8")))).convert("RGB")
    if should_prepare_image:
        image = prepare_image(image, size=(image_resolution, image_resolution))

    # Generate segmentation image
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)

    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if seed != None and seed != -1:
        generator = torch.Generator("cuda").manual_seed(seed)

    width, height = image.size

    # Run the model
    with autocast("cuda"):
        output = model(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            eta=eta,
            generator=generator,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        image = output.images[0]

    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
