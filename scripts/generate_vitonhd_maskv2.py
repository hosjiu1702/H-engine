import os
import argparse
from PIL import Image
from tqdm import trange, tqdm
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
    args = parser.parse_args()
    dataset_path = os.path.join(PROJECT_ROOT_PATH, args.vitonhd_path)
    image_path = {
        'train': os.path.join(dataset_path, 'train'),
        'test': os.path.join(dataset_path, 'test')
    }

    masker = Maskerv2()

    for mode, path in image_path.items():
        img_path = os.path.join(path, 'image')
        mask_path = os.path.join(path, 'agnostic-mask-v2')
        os.makedirs(mask_path, exist_ok=True)
        for fname in tqdm(os.listdir(img_path), ascii=True,
                            desc=f'{mode}', dynamic_ncols=True):
            fpath = os.path.join(img_path, fname)
            img = Image.open(fpath)
            mask = masker.create_mask(img)
            mask.save(os.path.join(mask_path, fname))


if __name__ == '__main__':
    main()