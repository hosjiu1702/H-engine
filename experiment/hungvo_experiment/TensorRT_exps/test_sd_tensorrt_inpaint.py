from stable_diffusion_tensorrt_inpaint import TensorRTStableDiffusionInpaintPipeline
from diffusers import StableDiffusionInpaintPipeline
import requests
from io import BytesIO
import PIL
from PIL import Image
from diffusers import PNDMScheduler, DPMSolverMultistepScheduler, DDIMScheduler, AutoencoderKL
import torch
from ip_adapter_tensorrt import IPAdapter, IPAdapterPlus
import os


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



def set_ip_adapter(pipe):
    #unet = pipe.unet
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipe.unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=4,
                ).to('cuda')
    pipe.unet.set_attn_processor(attn_procs)
    
ip_ckpt = "models/ip-adapter_sd15.bin"
image_encoder_path = "models/image_encoder/"

def load_ip_adapter(pipe):
    if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(ip_ckpt, map_location="cpu")
        #image_proj_model.load_state_dict(state_dict["image_proj"])
    ip_layers = torch.nn.ModuleList(pipe.unet.attn_processors.values())
    ip_layers.load_state_dict(state_dict["ip_adapter"])

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
    
    pipeline = TensorRTStableDiffusionInpaintPipeline.from_pretrained(
    "botp/stable-diffusion-v1-5-inpainting",
    #variant='fp16',
    vae = vae,
    safety_checker = None,
    torch_dtype=torch.float16,
    scheduler=scheduler,
    )
    
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
    
    # image = download_image(img_url).resize((512, 512))
    # mask_image = download_image(mask_url).resize((512, 512))
    
    pipeline = load_model()

    prompt = ""
    negative_prompt = ""
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    #image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask_image).images[0]
    
    image = pipeline.generate(prompt=prompt, negative_prompt=negative_prompt,pil_image=cloth_image, num_samples=1, num_inference_steps=100, seed=42, image=image, mask_image=mask_image)[0]
    
    image.save("./a_tensorrt.png")
    
    

    