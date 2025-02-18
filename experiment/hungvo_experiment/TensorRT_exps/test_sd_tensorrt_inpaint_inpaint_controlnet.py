from stable_diffusion_tensorrt_inpaint_controlnet import TensorRTStableDiffusionInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline
import requests
from io import BytesIO
import PIL
from PIL import Image
from diffusers import PNDMScheduler, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel
import torch
from ip_adapter_tensorrt_controlnet import IPAdapter, IPAdapterPlus

import os
from stable_diffusion_tensorrt_inpaint_controlnet import UNet2DConditionModelRT

from ip_adapter_src.utils import is_torch2_available, get_generator
from controlnet_aux import OpenposeDetector

if is_torch2_available():
    from ip_adapter_src.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from ip_adapter_src.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from ip_adapter_src.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter_src.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from ip_adapter_src.resampler import Resampler

ip_ckpt = "models/ip-adapter_sd15.bin"
image_encoder_path = "models/image_encoder/"


openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #scheduler = DPMSolverMultistepScheduler.from_pretrained("botp/stable-diffusion-v1-5-inpainting", subfolder="scheduler")
    # scheduler = DDIMScheduler(
    # num_train_timesteps=1000,
    # beta_start=0.00085,
    # beta_end=0.012,
    # beta_schedule="scaled_linear",
    # clip_sample=False,
    # set_alpha_to_one=False,
    # steps_offset=1,)
    
    
    #unet_org = UNet2DConditionModel.from_pretrained("botp/stable-diffusion-v1-5-inpainting", subfolder="unet", torch_dtype=torch.float16)
    #unet = UNet2DConditionModelRT(unet_org, controlnet)
    

    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    
#     pipeline = TensorRTStableDiffusionInpaintPipeline.from_pretrained(
#     "botp/stable-diffusion-v1-5-inpainting",
#     #variant='fp16',
#     vae = vae,
#     #unet = unet,
#     controlnet = None,
#     safety_checker = None,
#     torch_dtype=torch.float16,
#     scheduler=scheduler,
#     )
    

    ip_model = IPAdapter(image_encoder_path, ip_ckpt, device)
    
    
    return ip_model

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content))

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

if __name__ == '__main__':
    scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
    #scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler")
    
#     pipe = TensorRTStableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     #variant='fp16',
#     safety_checker = None,
#     torch_dtype=torch.float16,
#     scheduler=scheduler,
#     )

    cloth_image = Image.open("assets/clothes/images_1.jpg").convert("RGB")
    cloth_image.resize((256, 256))
    
    
    # image = download_image(img_url).resize((384, 384)).convert("RGB")
    # mask_image = download_image(mask_url).resize((384, 384)).convert("L")  
    
    # pipe.set_cached_folder("runwayml/stable-diffusion-inpainting")
    # pipe = pipe.to("cuda")
    
    image = Image.open("assets/inpainting/image1.png").resize((768, 1024)).convert("RGB")
    mask_image = Image.open("assets/inpainting/mask1.png").resize((768, 1024)).convert("L") 
    
    
    control_image = openpose(image)
    
    # image = download_image(img_url).resize((512, 512))
    # mask_image = download_image(mask_url).resize((512, 512))
    
    pipeline = load_model()

    prompt = ""
    negative_prompt = ""
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    #image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask_image).images[0]
    
    image = pipeline.generate(prompt=prompt, negative_prompt=negative_prompt,pil_image=cloth_image, num_samples=1, num_inference_steps=100, seed=42, image=image, mask_image=mask_image, control_image = control_image)[0]
    
    image.save("./a_tensorrt.png")
    
    

    