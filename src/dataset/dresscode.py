# Modified from https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

from typing import Text, Union, List, Dict
from pathlib import Path
import os
from os import path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from transformers import CLIPImageProcessor
from src.utils import is_image as is_valid

from IPython.core.debugger import Pdb


class DressCodeDataset(Dataset):

    def __init__(
            self,
            data_rootpath: Text,
            phase: Text, # train | test
            category: List[Text] = ['upper_body', 'lower_body', 'dresses'],
            order: Text = 'paired',
            use_augmentation: bool = False,
            h: int = 1024, # height
            w: int = 768, # weight
            use_dilated_relaxed_mask: bool = False,
    ):
        super(DressCodeDataset, self).__init__()
        self.data_rootpath = data_rootpath
        self.use_augmentation = use_augmentation
        self.h = h
        self.w = w
        self.use_dilated_relaxed_mask = use_dilated_relaxed_mask

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        im_names = []
        c_names = []
        dataroot_names = []

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.data_rootpath, c)
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
            """ Only apply this logic when the underlying mask generation process
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

        im_name = self.im_names[index]
        c_name = self.c_names[index]
        dataroot = self.dataroot_names[index]

        # person image
        img = Image.open(osp.join(dataroot, 'images', im_name))
        img = img.resize((self.w, self.h))
        img = self.transform(img)

        # garment image
        c = Image.open(osp.join(dataroot, 'images', c_name))
        c = c.resize((self.w, self.h))
        c = self.transform(c)

        # mask image
        mask = Image.open(osp.join(dataroot, 'mask_v2', im_name))
        mask = mask.resize((self.w, self.h))
        mask = mask.convert('L')
        mask = mask.point(lambda i: 255 if i > 127 else 0)
        mask = mask.convert('1') # [True, False] array
        mask = np.array(mask, dtype=np.float32) # [0, 1] array
        mask = torch.from_numpy(mask)

        # agnostic image (masked image)
        agn = Image.open(osp.join(dataroot, 'agnostic_v2', im_name))
        agn = agn.resize((self.w, self.h))
        agn = self.transform(agn)

        # skeleton image
        skl = Image.open(osp.join(dataroot, 'skeleton_modified', im_name))
        skl = skl.resize((self.w, self.h))
        skl = self.transform(skl)

        # densepose image
        dense = Image.open(osp.join(dataroot, 'dense_modified', im_name))
        dense = dense.resize((self.w, self.h))
        dense = self.transform(dense)

        item.update({
            'image': img,
            'masked_image': agn,
            'mask': mask,
            'densepose': dense,
            'cloth_raw': c,
        })

        return item