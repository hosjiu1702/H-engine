# Modified from https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

from typing import Text, Union, List, Dict
import random
from pathlib import Path
import os
from os import path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import adjust_hue, adjust_contrast
from PIL import Image
from transformers import CLIPImageProcessor
from src.utils import is_image as is_valid


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

        if self.use_augmentation:
            # flip
            self.flip = v2.RandomHorizontalFlip(p=1)
            # random shift
            shift_x = random.uniform(0, 0.2)
            shift_y = random.uniform(0, 0.2)
            self.random_shift = v2.RandomAffine(degrees=0, translate=(shift_x, shift_y))
            # random scale
            self.random_scale = v2.RandomAffine(degrees=0, scale=(0.8, 1.2))
            # hue adjustment
            self.hue_value = random.uniform(-0.5, 0.5)
            # contrast adjustment
            self.contrast_factor = random.uniform(0.8, 1.2)

        self.totensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
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
            """ Drop some images which are not existed in dilated mask folder for consistency when loading.
            This happens because our masker (v2) could not handle some hard cases from
            dataset so we ignore when preprocessing.
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

    def get_random_image(self):
        return self.get_random_sample()['image']

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        item = {}

        im_name = self.im_names[index]
        c_name = self.c_names[index]
        dataroot = self.dataroot_names[index]

        # person image
        img = Image.open(osp.join(dataroot, 'images', im_name))
        img = img.resize((self.w, self.h))
        clone_img = img.copy()
        img = self.transform(img)

        # garment image
        c_raw = Image.open(osp.join(dataroot, 'images', c_name))
        c_raw = c_raw.resize((self.w, self.h))
        c_raw = self.transform(c_raw)

        # mask image
        mask = Image.open(osp.join(dataroot, 'mask_v2', im_name))
        mask = mask.resize((self.w, self.h))
        mask = mask.convert('L')
        mask = mask.point(lambda i: 255 if i > 127 else 0)
        mask = mask.convert('1') # [True, False] array
        mask = np.array(mask, dtype=np.float32) # [0, 1] array
        mask = mask[None] # expand one more dimension for channel
        mask = torch.from_numpy(mask)

        # agnostic image (masked image)
        masked_img = Image.open(osp.join(dataroot, 'agnostic_v2', im_name))
        masked_img = masked_img.resize((self.w, self.h))
        masked_img = self.transform(masked_img)

        # skeleton image
        skl = Image.open(osp.join(dataroot, 'skeleton_modified', im_name))
        skl = skl.resize((self.w, self.h))
        skl = self.transform(skl)

        # densepose image
        dense = Image.open(osp.join(dataroot, 'dense_modified', im_name))
        dense = dense.resize((self.w, self.h))
        dense = self.transform(dense)

        if self.use_augmentation:
            if random.random() > 0.5:
                img = self.flip(img)
                c_raw = self.flip(c_raw)
                mask = self.flip(mask)
                masked_img = self.flip(masked_img)
                dp = self.flip(dp)
            if random.random() > 0.5:
                img = adjust_hue(img, self.hue_value)
                masked_img = adjust_hue(masked_img, self.hue_value)
                c_raw = adjust_hue(c_raw, self.hue_value)
            if random.random() > 0.5:
                img = adjust_contrast(img, self.contrast_factor)
                masked_img = adjust_contrast(masked_img, self.contrast_factor)
                c_raw = adjust_contrast(c_raw, self.contrast_factor)
            if random.random() > 0.5:
                img = self.random_shift(img)
                masked_img = self.random_shift(masked_img)
                mask = self.random_shift(mask)
                dp = self.random_shift(dp)
            if random.random() > 0.5:
                img = self.random_scale(img)
                masked_img = self.random_scale(masked_img)
                mask = self.random_scale(mask)
                dp = self.random_scale(dp)

        item.update({
            'im_name': im_name,
            'c_name': c_name,
            'original_image': self.totensor(clone_img),
            'image': img,
            'masked_image': masked_img,
            'mask': mask,
            'densepose': dense,
            'cloth_raw': c_raw,
        })

        return item

    def get_random_sample(self) -> Dict[Text, Image.Image]:
        idx = random.randint(0, len(self.im_names) - 1)

        im_name = self.im_names[idx]
        c_name = self.c_names[idx]
        dataroot = self.dataroot_names[idx]

        c = Image.open(osp.join(dataroot, 'images', c_name)).resize((self.w, self.h))
        img = Image.open(osp.join(dataroot, 'images', im_name)).resize((self.w, self.h))
        mask = Image.open(osp.join(dataroot, 'mask_v2', im_name)).resize((self.w, self.h))
        agn = Image.open(osp.join(dataroot, 'agnostic_v2', im_name)).resize((self.w, self.h))
        dense = Image.open(osp.join(dataroot, 'dense_modified', im_name)).resize((self.w, self.h))
        skl = Image.open(osp.join(dataroot, 'skeleton_modified', im_name)).resize((self.w, self.h))

        return {
            'cloth': c,
            'image': img,
            'mask': mask,
            'agnostic': agn,
            'dense': dense,
            'skeleton': skl
        }
