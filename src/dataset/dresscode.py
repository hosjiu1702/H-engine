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
from torchvision.transforms.functional import adjust_hue, adjust_contrast, affine, pil_to_tensor
from PIL import Image
from transformers import CLIPImageProcessor
from src.utils import is_image as is_valid, mask2agn, random_dilate_mask


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
            use_CLIPVision: bool = True,
            clip_model_id: str = 'openai/clip-vit-base-patch32', # huggingface model id
            use_dilated_relaxed_mask: bool = False,
            random_dilate_mask: bool = False
    ):
        super(DressCodeDataset, self).__init__()
        self.data_rootpath = data_rootpath
        self.use_augmentation = use_augmentation
        self.h = self.height = h
        self.w = self.width = w
        self.use_dilated_relaxed_mask = use_dilated_relaxed_mask
        self.random_dilate_mask = random_dilate_mask
        self.use_CLIPVision = use_CLIPVision

        if self.use_augmentation:
            # flip
            self.flip = v2.RandomHorizontalFlip(p=1)
        
        if self.use_CLIPVision:
            self.image_processor = CLIPImageProcessor(clip_model_id)

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
        c = c_raw = c_raw.resize((self.w, self.h))
        c = self.image_processor(images=c, return_tensors='pt').pixel_values
        c_raw = self.transform(c_raw)

        # mask image
        mask = Image.open(osp.join(dataroot, 'mask_v2', im_name))
        mask = mask.resize((self.w, self.h))
        
        ## In case mask values are not *real* binary ones.
        mask_np = np.array(mask)
        mask_np[mask_np > 127] = 255
        mask_np[mask_np <= 127] = 0
        mask = Image.fromarray(mask_np)

        if self.random_dilate_mask:
            if random.random() > 0.5:
                mask = random_dilate_mask(mask)

        mask = mask.convert('L')
        mask = mask.point(lambda i: 255 if i > 127 else 0)
        mask = mask.convert('1') # [True, False] array
        mask = np.array(mask, dtype=np.float32) # [0, 1] array
        mask = mask[None] # expand one more dimension for channel
        mask = torch.from_numpy(mask)

        # agnostic image (masked image)
        # masked_img = mask2agn(mask, clone_img)
        # masked_img = self.transform(masked_img)

        # skeleton image
        skl = Image.open(osp.join(dataroot, 'skeleton_modified', im_name))
        skl = skl.resize((self.w, self.h))
        skl = self.transform(skl)

        # densepose image
        dp = Image.open(osp.join(dataroot, 'dense_modified', im_name))
        dp = dp.resize((self.w, self.h))
        dp = self.transform(dp)

        if self.use_augmentation:
            if random.random() > 0.5:
                img = self.flip(img)
                c_raw = self.flip(c_raw)
                c = self.flip(c)
                mask = self.flip(mask)
                dp = self.flip(dp)
            if random.random() > 0.5:
                hue_value = random.uniform(-0.5, 0.5)
                img = adjust_hue(img, hue_value)
                c_raw = adjust_hue(c_raw, hue_value)
                c = adjust_hue(c, hue_value)
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                img = adjust_contrast(img, contrast_factor)
                c_raw = adjust_contrast(c_raw, contrast_factor)
                c = adjust_contrast(c, contrast_factor)
            # if random.random() > 0.5:
            #     shift_x = random.uniform(-0.2, 0.2)
            #     shift_y = random.uniform(-0.2, 0.2)
            #     img = affine(img, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
            #     masked_img = affine(masked_img, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
            #     mask = affine(mask, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
            #     dp = affine(dp, angle=0, translate=(shift_x * self.width, shift_y * self.height), scale=1, shear=0)
            # if random.random() > 0.5:
            #     scale = random.uniform(0.8, 1.2)
            #     img = affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)
            #     # masked_img = affine(masked_img, angle=0, translate=(0, 0), scale=scale, shear=0)
            #     mask = affine(mask, angle=0, translate=(0, 0), scale=scale, shear=0)
            #     dp = affine(dp, angle=0, translate=(0, 0), scale=scale, shear=0)
        
        masked_img = torch.mul(1 - mask, img)
        
        drop_image_embed = 1 if random.random() < 0.1 else 0

        item.update({
            'im_name': im_name,
            'c_name': c_name,
            'original_image': self.totensor(clone_img),
            'image': img,
            'masked_image': masked_img,
            'mask': mask,
            'densepose': dp,
            'cloth_raw': c_raw,
            'cloth_ref': c,
            'drop_image_embed': drop_image_embed
        })

        return item

    def get_random_sample(self) -> Dict[Text, Image.Image]:
        idx = random.randint(0, len(self.im_names) - 1)

        im_name = self.im_names[idx]
        c_name = self.c_names[idx]
        dataroot = self.dataroot_names[idx]

        c_raw = Image.open(osp.join(dataroot, 'images', c_name)).resize((self.w, self.h))
        img = Image.open(osp.join(dataroot, 'images', im_name)).resize((self.w, self.h))
        mask = Image.open(osp.join(dataroot, 'mask_v2', im_name)).resize((self.w, self.h))
        masked_img = mask2agn(pil_to_tensor(mask), img)
        dp = Image.open(osp.join(dataroot, 'dense_modified', im_name)).resize((self.w, self.h))
        skl = Image.open(osp.join(dataroot, 'skeleton_modified', im_name)).resize((self.w, self.h))

        return {
            'cloth': c_raw,
            'image': img,
            'mask': mask,
            'agnostic': masked_img,
            'dense': dp,
            'skeleton': skl
        }
