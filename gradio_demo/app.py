from os import path as osp
from enum import Enum
import PIL
from PIL import Image, ImageOps
import gradio as gr
import torch
from torchvision.transforms.functional import pil_to_tensor
from diffusers import DiffusionPipeline
from src.pipelines.spacat_pipeline import TryOnPipeline
from src.models.utils import download_model, load_model, get_densepose_map, preprocess_image
from src.utils.mask_v2 import Maskerv2 as Masker
from src.utils import get_project_root


PROJECT_ROOT_PATH = get_project_root()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GradioImageSize(Enum):
    WIDTH = 512
    HEIGHT = 512


class OutputSize(Enum):
    W = 384
    H = 512


masker = Masker()

# Download model
model_path = download_model(
    repo_id='bui/Navier-1',
    ckpt_name='ckpt-20000-1512-preview',
    model_name='Navier-1'
)
unet, vae, scheduler = load_model(model_path)

# Load diffusion try-on pipeline
pipe = TryOnPipeline(
    unet=unet,
    vae=vae,
    scheduler=scheduler
).to(device)


def try_on(pipeline: DiffusionPipeline, person_img_path: str, garment_img_path: str, masker):
    """
    Main function to run try-on process.
    
    Args:
        pipeline (DiffusionPipeline): Diffusion pipeline to do the try-on task.
        person_img (str): path to the person image.
        garment_img (str): path to the garment image.
    
    Returns:
        Try-on image
    """
    # Load images
    person = Image.open(person_img_path)
    garment = Image.open(garment_img_path)

    # Resize images to the allowed resolution
    person = ImageOps.fit(person, size=(OutputSize.W, OutputSize.H))
    garment = ImageOps.fit(garment, size=(OutputSize.W, OutputSize.H))

    # Get mask
    mask = masker.create_mask(person_img)

    # Get densepose
    densepose = get_densepose_map(person_img_path, size=(OutputSize.W, OutputSize.H))

    # Preprocessing
    person = preprocess_image(person, OutputSize.W, OutputSize.H).unsqueeze(0)
    garment = preprocess_image(garment, OutputSize.W, OutputSize.H).unsqueeze(0)
    mask = pil_to_tensor(mask).unsqueeze(0)
    densepose = pil_to_tensor(densepose).unsqueeze(0)

    with torch.inference_mode():
        with torch.amp.autocast(device):
            image = pipeline(
                image=person.to(device),
                mask_image=mask.to(device),
                densepose_image=densepose.to(device),
                cloth_image=garment.to(device),
                height=OutputSize.H,
                width=OutputSize.W,
                generator=torch.manual_seed(1996),
                guidance_scale=1.5,
            ).image[0]
    
    return image


with gr.Blocks(theme='ParityError/Interstellar') as demo:
    title = "## Heatmob Virtual Try-on Demo."
    gr.Markdown(title)
    gr.Markdown(f"**Model version:** *Navier-1[Beta]-1512-preview*")
    with gr.Row():
        with gr.Column():
            gr.Markdown('#### Person Image')
            person_img = gr.Image(
                sources=['upload', 'webcam'],
                type='filepath',
                width=GradioImageSize.WIDTH,
                height=GradioImageSize.HEIGHT,
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
            gr.Markdown('#### Garment Image')
            garment_img = gr.Image(
                sources=['upload', 'webcam'],
                type='filepath',
                width=GradioImageSize.WIDTH,
                height=GradioImageSize.HEIGHT
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
            gr.Markdown('#### Output')
            generated_img = gr.Image(
                width=GradioImageSize.WIDTH,
                height=GradioImageSize.HEIGHT,
                interactive=False
            )
            with gr.Row():
                tryon_button = gr.Button('Press to try-on')
                tryon_button.click(fn=try_on, inputs=[person_img, garment_img], outputs=[generated_img])


if __name__ == "__main__":
    demo.launch()