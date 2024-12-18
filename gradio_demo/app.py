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
from src.utils import get_project_root


PROJECT_ROOT_PATH = get_project_root()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['PYTHONBREAKPOINT'] = "0"
load_dotenv()


class OutputSize(Enum):
    W = 384
    H = 512


masker = Masker()

# Download model
model_path = download_model(
    repo_id='bui/Navier-1',
    ckpt_name='ckpt-20000-1512-preview',
    model_name='Navier-1',
    token=os.getenv('HF_TOKEN')
)
unet, vae, scheduler = load_model(model_path)

# Load diffusion try-on pipeline
pipeline = TryOnPipeline(
    unet=unet,
    vae=vae,
    scheduler=scheduler
).to(device)


def try_on(person_img_path: str, garment_img_path: str, poisson_blending: bool):
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
    mask = masker.create_mask(person)

    # Get densepose
    densepose = get_densepose_map(person_img_path, size=(OutputSize.W.value, OutputSize.H.value))

    # Preprocessing
    person_tensor = preprocess_image(person, OutputSize.W.value, OutputSize.H.value).unsqueeze(0)
    garment_tensor = preprocess_image(garment, OutputSize.W.value, OutputSize.H.value).unsqueeze(0)
    mask_tensor = pil_to_tensor(mask).unsqueeze(0)
    densepose_tensor = pil_to_tensor(densepose).unsqueeze(0)

    with torch.inference_mode():
        with torch.amp.autocast(device):
            tryon = pipeline(
                image=person_tensor.to(device),
                mask_image=mask_tensor.to(device),
                densepose_image=densepose_tensor.to(device),
                cloth_image=garment_tensor.to(device),
                height=OutputSize.H.value,
                width=OutputSize.W.value,
                generator=torch.manual_seed(1996),
                guidance_scale=1.5,
            ).images[0]
    
    if poisson_blending:
        tryon = apply_poisson_blending(person, tryon, mask)

    return tryon


with gr.Blocks(theme='ParityError/Interstellar').queue(max_size=10) as demo:
    title = "## Heatmob Virtual Try-on Demo ♨️"
    gr.Markdown(title)
    gr.Markdown(f"**👷Model version:** Navier-1[Beta]-1512-preview")
    gr.Markdown(f'**🗂️Supported Category:** Upper Garment')
    gr.Markdown(f'**🖥️Supported Resolution:** 384x512 *(output)*')
    gr.Markdown(f'**💾Training Dataset:** VITON-HD *(downscaled)*')
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
                examples_per_page=5,
                examples=[
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00064_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00205_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00272_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00396_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/person/00458_00.jpg')
                ],
            )

        with gr.Column():
            garment_img = gr.Image(
                sources=['upload', 'webcam'],
                label='Garment',
                type='filepath',
                interactive=True
            )
            gr.Examples(
                inputs=garment_img,
                examples_per_page=5,
                examples=[
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00205_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00311_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00339_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/00641_00.jpg'),
                    osp.join(PROJECT_ROOT_PATH, 'gradio_demo/examples/garment/01048_00.jpg')
                ],
            )

        with gr.Column():
            generated_img = gr.Image(
                label='Output',
                interactive=False
            )
            with gr.Row():
                tryon_button = gr.Button('Press to try-on')
            with gr.Row():
                poisson_blending = gr.Checkbox(label='Poisson Blending', info='Image Enhancer (post-processing)')

        tryon_button.click(
            fn=try_on,
            inputs=[person_img, garment_img, poisson_blending],
            outputs=[generated_img]
        )


if __name__ == "__main__":
    demo.launch(share=True, auth=('heatmob', 'navier'))