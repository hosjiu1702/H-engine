from dataclasses import dataclass
from typing import Text, Union, List
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from transformers import CLIPImageProcessor


@dataclass
class DownscaleResolution: 
    resolution = (512, 384)


class VITONHDDataset(Dataset):

    def __init__(
            self,
            data_rootpath: Text,
            use_trainset: bool = True,
            use_paired_data: bool = True,
            use_augmentation: bool = False,
            height: int = 1024,
            width: int = 768,
            use_CLIPVision: bool = True,
            downscale: bool = False
    ):
        super(VITONHDDataset, self).__init__()
        self.data_rootpath = data_rootpath
        self.use_trainset = use_trainset
        self.use_paired_data = use_paired_data
        self.use_augmentation = use_augmentation
        self.height = height
        self.width = width
        self.use_CLIPVision = use_CLIPVision
        self.downscale = downscale

        self.totensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        if use_CLIPVision:
            self.image_processor = CLIPImageProcessor()

        mode = 'train' if self.use_trainset else 'test'
        datapath = os.path.join(data_rootpath, mode)

        def _get_file_paths(root_dir: Union[Path, Text]) -> List[Path]:
            return [Path(root_dir, fname) for fname in os.listdir(root_dir)]

        if self.use_paired_data:
            self.im_paths = _get_file_paths(Path(datapath, 'image')) # person image
            self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask')) # mask (outfit)
            self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v3.2')) # masked person image
            self.c_paths = _get_file_paths(Path(datapath, 'cloth')) # outfit image
            self.dp_paths = _get_file_paths(Path(datapath, 'image-densepose')) # densepose image
        else:
            raise ValueError('Not support unpaired setting for VITON-HD dataset yet.')

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        item = {}

        img = Image.open(self.im_paths[index])
        c = Image.open(self.c_paths[index])
        dp = Image.open(self.dp_paths[index])
        mask = Image.open(self.m_paths[index]).convert('L')
        masked_img = Image.open(self.agn_paths[index])

        if self.downscale:
            new_size = DownscaleResolution.resolution

            img = img.resize(new_size)
            c = c.resize(new_size)
            dp = dp.resize(new_size)
            mask = mask.resize(new_size)
            masked_img = masked_img.resize(new_size)

            self.height = new_size[0]
            self.width = new_size[1]

        # Person image
        origin_img = img = img.resize((self.width, self.height))
        img = self.transform(img)

        # Cloth
        c = self.image_processor(images=c, return_tensors='pt').pixel_values
        c = c.squeeze(0)
        
        # Densepose
        dp = dp.resize((self.width, self.height))
        dp = self.transform(dp)

        # Mask
        mask = mask.resize((self.width, self.height))
        mask = self.totensor(mask)
        # A dirty snippet code that check if this is a binary matrix
        # uni_elems = mask.unique()
        # assert 1. in uni_elems
        # assert 0. in uni_elems
        # assert len(uni_elems) == 2

        # Masked image (agnostic image)
        masked_img = masked_img.resize((self.width, self.height))
        masked_img = self.transform(masked_img)

        item.update({
            'original_image': self.totensor(origin_img),
            'image': img,
            'masked_image': masked_img,
            'mask': mask,
            'densepose': dp,
            'cloth': c
        })

        return item