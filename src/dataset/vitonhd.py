import random
from typing import Text, Union, List
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import adjust_hue, adjust_contrast, affine
from PIL import Image
from transformers import CLIPImageProcessor
from src.utils import is_image, random_dilate_mask, mask2agn


class VITONHDDataset(Dataset):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    def __init__(
            self,
            data_rootpath: Text,
            use_trainset: bool = True,
            use_paired_data: bool = True,
            use_augmentation: bool = False,
            random_dilate_mask: bool = False,
            height: int = 1024,
            width: int = 768,
            use_CLIPVision: bool = True,
            use_dilated_relaxed_mask: bool = False,
    ):
        super(VITONHDDataset, self).__init__()
        self.data_rootpath = data_rootpath
        self.use_trainset = use_trainset
        self.use_paired_data = use_paired_data
        self.use_augmentation = use_augmentation
        self.height = height
        self.width = width
        self.use_CLIPVision = use_CLIPVision
        self.use_dilated_relaxed_mask = use_dilated_relaxed_mask
        self.random_dilate_mask = random_dilate_mask

        if self.use_augmentation:
            # flip
            self.flip = v2.RandomHorizontalFlip(p=1)

        self.totensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        if use_CLIPVision:
            self.image_processor = CLIPImageProcessor()

        mode = 'train' if self.use_trainset else 'test'
        datapath = os.path.join(data_rootpath, mode)

        def _get_file_paths(root_dir: Union[Path, Text]) -> List[Path]:
            return [Path(root_dir, fname) for fname in sorted(os.listdir(root_dir)) if is_image(fname)]

        if mode == 'train':
            if self.use_paired_data:
                self.im_paths = _get_file_paths(Path(datapath, 'image'))
                if self.use_dilated_relaxed_mask:
                    self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask-v2'))
                    self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v2'))
                else:
                    self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask'))
                    self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v3.2'))
                self.c_paths = _get_file_paths(Path(datapath, 'cloth'))
                self.dp_paths = _get_file_paths(Path(datapath, 'image-densepose'))
            else:
                raise ValueError('Not support unpaired setting for VITON-HD dataset yet.')
        else:
            self.im_paths = _get_file_paths(Path(datapath, 'image'))
            if self.use_dilated_relaxed_mask:
                self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask-v2'))
                self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v2'))
            else:
                self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask'))
                self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v3.2'))
            self.c_paths = _get_file_paths(Path(datapath, 'cloth'))
            self.dp_paths = _get_file_paths(Path(datapath, 'image-densepose'))

    def get_random_image(self):
        img_path = random.choice(self.im_paths)
        img = Image.open(img_path)
        return img

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        item = {}

        img_name = str(self.im_paths[index]).split('/')[-1]

        # Person image
        img = Image.open(self.im_paths[index])
        origin_img = img = img.resize((self.width, self.height))
        img = self.transform(img)

        # Cloth
        c = Image.open(self.c_paths[index])
        c = self.image_processor(images=c, return_tensors='pt').pixel_values
        c = c.squeeze(0)

        # Cloth (no CLIP preprocessing)
        c_raw = Image.open(self.c_paths[index])
        c_raw = c_raw.resize((self.width, self.height))
        c_raw = self.transform(c_raw)

        # Densepose
        dp = Image.open(self.dp_paths[index])
        origin_dp = dp = dp.resize((self.width, self.height))
        dp = self.transform(dp)

        # Mask
        mask = Image.open(self.m_paths[index])
        origin_m = mask = mask.resize((self.width, self.height))

        ## In case mask values are not *real* binary ones.
        mask_np = np.array(mask)
        mask_np[mask_np > 127] = 255
        mask_np[mask_np <= 127] = 0
        mask = Image.fromarray(mask_np)
        
        mask = random_dilate_mask(mask) if self.random_dilate_mask else mask
        mask = self.totensor(mask.convert('L'))
        mask[mask>0.5] = 1.
        mask[mask<0.5] = 0.

        # Masked image (agnostic image)
        masked_img = mask2agn(mask, origin_img)
        masked_img = self.transform(masked_img)

        if self.use_augmentation:
            if random.random() > 0.5:
                img = self.flip(img)
                c_raw = self.flip(c_raw)
                mask = self.flip(mask)
                masked_img = self.flip(masked_img)
                dp = self.flip(dp)
            if random.random() > 0.5:
                hue_value = random.uniform(-0.5, 0.5)
                img = adjust_hue(img, hue_value)
                masked_img = adjust_hue(masked_img, hue_value)
                c_raw = adjust_hue(c_raw, hue_value)
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                img = adjust_contrast(img, contrast_factor)
                masked_img = adjust_contrast(masked_img, contrast_factor)
                c_raw = adjust_contrast(c_raw, contrast_factor)
            if random.random() > 0.5:
                shift_x = random.uniform(-0.2, 0.2)
                shift_y = random.uniform(-0.2, 0.2)
                img = affine(img, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
                masked_img = affine(masked_img, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
                mask = affine(mask, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
                dp = affine(dp, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2)
                img = affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)
                masked_img = affine(masked_img, angle=0, translate=(0, 0), scale=scale, shear=0)
                mask = affine(mask, angle=0, translate=(0, 0), scale=scale, shear=0)
                dp = affine(dp, angle=0, translate=(0, 0), scale=scale, shear=0)

        item.update({
            'im_name': img_name,
            'c_name': '',
            'original_image': self.totensor(origin_img),
            'image': img,
            'masked_image': masked_img,
            'mask': mask,
            'densepose': dp,
            'cloth_raw': c_raw,
            # 'original_image_path': str(self.im_paths[index]),
            # 'original_mask': self.totensor(origin_m),
            # 'original_mask_path': str(self.m_paths[index]),
            # 'original_masked_image': self.totensor(origin_agn),
            # 'original_densepose': self.totensor(origin_dp),
            # 'original_cloth_path': str(self.c_paths[index]),
            # 'cloth': c,
        })

        return item

    @classmethod
    def preprocess(cls, img: Image.Image, width: int, height: int) -> torch.Tensor:
        x = img.resize((width, height))
        x = cls.transform(x)
        return x
