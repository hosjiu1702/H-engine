# Modified from https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

from typing import Text, Union, List, Dict
from pathlib import Path
import os
from os import path as osp
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from transformers import CLIPImageProcessor
from src.utils import is_image as is_valid


class DressCodeDataset(Dataset):

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
            phase: Text, # train | test
            category: List[Text] = ['upper_body', 'lower_body', 'dresses'],
            order: Text = 'paired',
            use_augmentation: bool = False,
            height: int = 1024,
            width: int = 768,
            use_dilated_relaxed_mask: bool = False,
    ):
        super(DressCodeDataset, self).__init__()
        self.data_rootpath = data_rootpath
        self.use_augmentation = use_augmentation
        self.height = height
        self.width = width
        self.use_dilated_relaxed_mask = use_dilated_relaxed_mask

        im_names = []
        c_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")
            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        if use_dilated_relaxed_mask:
            """ Apply when the underlying mask generation process
            could not handle all of inputs from images/ folder.
            """
            drop_indices = []
            for i, rootpath in enumerate(dataroot_names):
                query_name = im_names[i]
                mask_path = osp.join(rootpath, 'mask_v2', query_name)
                if not osp.isfile(mask_path):
                    drop_indices.append(i)
            tmp_im = [im_name for idx, im_name in enumerate(im_names) if idx not in drop_indices]
            tmp_c = [c_name for idx, c_name in enumerate(c_names) if idx not in drop_indices]
            tmp_dataroot = [dataroot_name for idx, dataroot_name in enumerate(dataroot_names) if idx not in drop_indices]
            im_names = tmp_im
            c_names = tmp_c
            dataroot_names = tmp_dataroot
        
        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

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

        # Densepose
        dp = Image.open(self.dp_paths[index])
        origin_dp = dp = dp.resize((self.width, self.height))
        dp = self.transform(dp)

        # Mask
        mask = Image.open(self.m_paths[index])
        origin_m = mask = mask.resize((self.width, self.height))
        mask = self.totensor(mask.convert('L'))

        # Masked image (agnostic image)
        masked_img = Image.open(self.agn_paths[index])
        origin_agn = masked_img = masked_img.resize((self.width, self.height))
        masked_img = self.transform(masked_img)

        item.update({
            'original_image': self.totensor(origin_img),
            'original_image_path': str(self.im_paths[index]),
            'original_mask': self.totensor(origin_m),
            'original_mask_path': str(self.m_paths[index]),
            'original_masked_image': self.totensor(origin_agn),
            'original_densepose': self.totensor(origin_dp),
            'original_cloth_path': str(self.c_paths[index]),
            'image': img,
            'masked_image': masked_img,
            'mask': mask,
            'densepose': dp,
            'cloth_raw': c_raw,
            'cloth': c,
            'im_name': img_name
        })

        return item