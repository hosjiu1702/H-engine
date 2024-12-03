import os
import argparse
from PIL import Image
from tqdm.auto import trange, tqdm
import numpy as np
from einops import rearrange
from src.utils.mask_v2 import Maskerv2
from src.utils import get_project_root


PROJECT_ROOT_PATH = get_project_root()


def main():
    parser = argparse.ArgumentParser(
        description='Script to generate mask v2 for vitonhd dataset'
    )
    parser.add_argument(
        '--vitonhd_path',
        type=str,
        required=True,
        help='Path to the VITON-HD dataset.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='agnostic-mask-v2'
    )
    args = parser.parse_args()
    dataset_path = os.path.join(PROJECT_ROOT_PATH, args.vitonhd_path)
    image_path = {
        'train': os.path.join(dataset_path, 'train'),
        'test': os.path.join(dataset_path, 'test')
    }

    masker = Maskerv2()

    for mode, path in tqdm(image_path.items()):
        img_path = os.path.join(path, 'image')
        mask_path = os.path.join(path, args.output_dir)
        masked_img_path = os.path.join(path, 'agnostic-v2')
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(masked_img_path, exist_ok=True)
        for fname in tqdm(os.listdir(img_path), desc=mode):
            fpath = os.path.join(img_path, fname)
            img = Image.open(fpath)
            mask = None
            saved_path = os.path.join(mask_path, fname)
            if not os.path.isfile(saved_path):
                mask = masker.create_mask(img)
                mask.save(saved_path)
            mask = mask if mask else Image.open(saved_path)
            mask_np = np.array(mask)
            mask_np = np.stack([mask_np] * 3)
            mask_np = rearrange(mask_np, 'c h w -> h w c')
            img_np = np.array(img)
            masked_img_np = np.where(mask_np, np.ones_like(mask_np) * 127, img_np)
            masked_img = Image.fromarray(masked_img_np)
            saved_path = os.path.join(masked_img_path, fname)
            masked_img.save(saved_path)


if __name__ == '__main__':
    main()