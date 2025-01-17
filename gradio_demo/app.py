import os
from os import path as osp
from enum import Enum
from dotenv import load_dotenv
import PIL
from PIL import Image, ImageOps
import gradio as gr
import torch
from torchvision.transforms.functional import pil_to_tensor
from src.pipelines.spacat_pipeline import TryOnPipeline
from src.models.utils import download_model, load_model, get_densepose_map, preprocess_image, apply_poisson_blending
from src.utils.mask_v2 import Maskerv2 as Masker
from src.utils import get_project_root, mask2agn


PROJECT_ROOT_PATH = get_project_root()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['PYTHONBREAKPOINT'] = "0"
load_dotenv()


class OutputSize(Enum):
    W = 600
    H = 800


masker = Masker()

# Download model
# model_path = download_model(
#     repo_id='bui/Navier-1',
#     ckpt_name='ckpt-20000-1512-preview',
#     model_name='Navier-1',
#     token=os.getenv('HF_TOKEN')
# )
model_path = osp.join(PROJECT_ROOT_PATH, 'checkpoints/navier-1/navier-1-beta-1512-800x600/310000-steps')
unet, vae, scheduler = load_model(model_path)

# Load diffusion try-on pipeline
pipeline = TryOnPipeline(
    unet=unet,
    vae=vae,
    scheduler=scheduler
).to(device)


def try_on(
        person_img_path: str,
        garment_img_path: str,
        poisson_blending: bool,
        category: str,
        model_name: str,
        inference_steps: int
):
    """
    Main function to run try-on process.
    
    Args:
        person_img_path (str): path to the person image.
        garment_img_path (str): path to the garment image.
        poisson_blending (bool): whether or not to use Poisson Image Blending.

    Returns:
        Try-on image
    """
    # Load images
    person = Image.open(person_img_path)
    garment = Image.open(garment_img_path)

    # Resize images to the allowed resolution
    person = ImageOps.fit(person, size=(OutputSize.W.value, OutputSize.H.value))
    garment = ImageOps.fit(garment, size=(OutputSize.W.value, OutputSize.H.value))

    # Get mask
    if category == 'overall':
        category = 'dresses'
    else:
        category += '_body'
    mask = masker.create_mask(person, category)

    # Get densepose
    densepose = get_densepose_map(person_img_path, size=(OutputSize.W.value, OutputSize.H.value))

    # Preprocessing
    person_tensor = preprocess_image(person, OutputSize.W.value, OutputSize.H.value).unsqueeze(0)
    garment_tensor = preprocess_image(garment, OutputSize.W.value, OutputSize.H.value).unsqueeze(0)
    mask_tensor = pil_to_tensor(mask).unsqueeze(0)
    densepose_tensor = pil_to_tensor(densepose).unsqueeze(0)

    with torch.inference_mode():
        with torch.amp.autocast(device):
            # pipe = pipeline if model_name == '1512' else pipeline2
            tryon = pipeline(
                image=person_tensor.to(device),
                mask_image=mask_tensor.to(device),
                densepose_image=densepose_tensor.to(device),
                cloth_image=garment_tensor.to(device),
                height=OutputSize.H.value,
                width=OutputSize.W.value,
                generator=torch.manual_seed(1996),
                guidance_scale=1.5,
                num_inference_steps=inference_steps
            ).images[0]
    
    if poisson_blending:
        tryon = apply_poisson_blending(person, tryon, mask)

    return tryon


with gr.Blocks(theme='ParityError/Interstellar').queue(max_size=10) as demo:
    title = "## Heatmob Virtual Try-on Demo ♨️"
    gr.Markdown(title)
    gr.Markdown(f"**👷Model version:** Navier-1[Beta].1512")
    gr.Markdown(f'**🗂️Supported Category:** Upper-Body, Lower-Body, Dresses')
    gr.Markdown(f'**🖥️Supported Resolution:** 800x600 *(output)*')
    gr.Markdown(f'**💾Training Dataset:** VITON-HD & DressCode')
    with gr.Row():
        with gr.Column():
            person_img = gr.Image(
                sources=['upload', 'webcam'],
                label='Person',
                type='filepath',
                interactive=True,
            )
            gr.Examples(
                inputs=person_img,
                examples_per_page=15,
                examples=[
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00064_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00205_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00272_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00396_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00458_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full1.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full2.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full3.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full4.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full5.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full6.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full7.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full8.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full9.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/full10.jpg')
                ],
            )

        with gr.Column():
            garment_img = gr.Image(
                sources=['upload', 'webcam'],
                label='Garment',
                type='filepath',
                interactive=True
            )
            category = gr.Radio(choices=['upper', 'lower', 'overall'], label='Garment Category', value='upper')
            gr.Examples(
                inputs=garment_img,
                examples_per_page=15,
                examples=[
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00205_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00311_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00339_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00641_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/01048_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/dress1.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/dress2.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/dress3.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/dress4.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/dress5.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/combo1.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/combo2.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/combo3.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/combo4.png'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/combo5.png'),
                ],
            )
        
        with gr.Column():
            with gr.Row():
                generated_mask = gr.Image(label='Mask', interactive=False)
                generated_img = gr.Image(label='Output', interactive=False)
            
            model_name = gr.Dropdown(['1512[800x600]'], value='1512[800x600]', label='Models', info='')
            mask_btn = gr.Button('Step 1: Run Mask')
            tryon_btn = gr.Button('Step 2: Try-on')
            poisson_blending = gr.Checkbox(value=True, label='Poisson Blending', info='Image Enhancer (post-processing)')

            with gr.Accordion('Advanced Options', open=False):
                inference_steps = gr.Slider(minimum=10, maximum=50, value=40, step=5, label='Inference Steps')
            
            def _get_mask(img_path, ctg):
                img = Image.open(img_path)
                img = ImageOps.fit(img, size=(OutputSize.W.value, OutputSize.H.value))
                if ctg == 'overall': 
                    ctg = 'dresses'
                else:
                    ctg += '_body'
                mask = masker.create_mask(img, ctg, return_img=False)
                return mask2agn(mask, img)

            mask_btn.click(
                fn=_get_mask,
                inputs=[person_img, category],
                outputs=[generated_mask]
            )
            tryon_btn.click(
                fn=try_on,
                inputs=[person_img, garment_img, poisson_blending, category, model_name, inference_steps],
                outputs=[generated_img]
            )


if __name__ == "__main__":
    demo.launch(share=True)