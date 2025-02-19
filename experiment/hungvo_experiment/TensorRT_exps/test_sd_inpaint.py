from stable_diffusion_tensorrt_inpaint import TensorRTStableDiffusionInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline
import requests
from io import BytesIO
import PIL
from PIL import Image
from diffusers import PNDMScheduler, DPMSolverMultistepScheduler, DDIMScheduler
import torch
from ip_adapter import IPAdapter, IPAdapterPlus
import os
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
from stable_diffusion_tensorrt_inpaint import StableDiffusionControlNetInpaintPipeline
import numpy as np


from ip_adapter_src.utils import is_torch2_available, get_generator

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


openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)



ip_ckpt = "models/ip-adapter_sd15.bin"
image_encoder_path = "models/image_encoder/"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #scheduler = DPMSolverMultistepScheduler.from_pretrained("botp/stable-diffusion-v1-5-inpainting", subfolder="scheduler")
    scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "botp/stable-diffusion-v1-5-inpainting",
    #variant='fp16',
    vae = vae, 
    controlnet=controlnet,
    safety_checker = None,
    torch_dtype=torch.float16,
    scheduler=scheduler,
    )
    
    # pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    # #variant='fp16',
    # vae = vae, 
    # #controlnet=controlnet,
    # safety_checker = None,
    # torch_dtype=torch.float16,
    # scheduler=scheduler,
    # )
    

#     set_ip_adapter(pipeline)
#     load_ip_adapter(pipeline)


    # pipeline.set_cached_folder("runwayml/stable-diffusion-inpainting", variant='fp16')
    # pipeline = pipeline.to("cuda")
    
    
    # pipeline.unet = torch.compile(pipeline.unet)  # Compile the U-Net
    # pipeline.vae = torch.compile(pipeline.vae)    # Compile the VAE
    # pipeline.text_encoder = torch.compile(pipeline.text_encoder)
    
    ip_model = IPAdapter(pipeline, image_encoder_path, ip_ckpt, device)
    
    #ip_model = ip_model.to(device)
    
    return ip_model





def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content))

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

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
    cloth_image.resize((512, 512))
    
    
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
    
    image = pipeline.generate(pil_image=cloth_image, num_samples=4, num_inference_steps=25,
                           seed=42, image=image, mask_image=mask_image, control_image = control_image, strength=1.0, height = 1024, width = 768)[0]
    
    image.save("./a.png")

    

    
