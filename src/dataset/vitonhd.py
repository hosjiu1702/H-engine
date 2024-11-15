from typing import Text, Union, List
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from transformers import CLIPImageProcessor


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
            height: int = 1024,
            width: int = 768,
            use_CLIPVision: bool = True,
    ):
        super(VITONHDDataset, self).__init__()
        self.data_rootpath = data_rootpath
        self.use_trainset = use_trainset
        self.use_paired_data = use_paired_data
        self.use_augmentation = use_augmentation
        self.height = height
        self.width = width
        self.use_CLIPVision = use_CLIPVision

        self.totensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        if use_CLIPVision:
            self.image_processor = CLIPImageProcessor()

        mode = 'train' if self.use_trainset else 'test'
        datapath = os.path.join(data_rootpath, mode)

        def _get_file_paths(root_dir: Union[Path, Text]) -> List[Path]:
            return [Path(root_dir, fname) for fname in sorted(os.listdir(root_dir))]

        if mode == 'train':
            if self.use_paired_data:
                self.im_paths = _get_file_paths(Path(datapath, 'image')) # person image
                self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask')) # mask (outfit)
                self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v3.2')) # masked person image
                self.c_paths = _get_file_paths(Path(datapath, 'cloth')) # outfit image
                self.dp_paths = _get_file_paths(Path(datapath, 'image-densepose')) # densepose image
            else:
                raise ValueError('Not support unpaired setting for VITON-HD dataset yet.')
        else:
            self.im_paths = _get_file_paths(Path(datapath, 'image')) # person image
            self.m_paths = _get_file_paths(Path(datapath, 'agnostic-mask')) # mask (outfit)
            self.agn_paths = _get_file_paths(Path(datapath, 'agnostic-v3.2')) # masked person image
            self.c_paths = _get_file_paths(Path(datapath, 'cloth')) # outfit image
            self.dp_paths = _get_file_paths(Path(datapath, 'image-densepose')) # densepose image

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
        origin_dp = dp.resize((self.width, self.height))
        dp = self.transform(dp)

        # Mask
        mask = Image.open(self.m_paths[index])
        origin_m = mask = mask.resize((self.width, self.height))
        mask = self.totensor(mask.convert('L'))
        # A dirty snippet code that check if this is a binary matrix
        # uni_elems = mask.unique()
        # assert 1. in uni_elems
        # assert 0. in uni_elems
        # assert len(uni_elems) == 2

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

    @classmethod
    def preprocess(cls, img: Image.Image, width: int, height: int) -> torch.Tensor:
        x = img.resize((width, height))
        x = cls.transform(x)
        return x
