#!/bin/python
from typing import Text
import os
from os import path as osp
from PIL import Image
from tqdm import tqdm
import fire
import numpy as np
from einops import rearrange
# from joblib import Parallel, delayed, wrap_non_picklable_objects
from src.utils.mask_v2 import Maskerv2 as Masker
from src.models.utils import get_densepose_map
from src.preprocess.openpose import OpenPose


os.environ['PYTHONBREAKPOINT'] = '0'
IMAGE_INDICATOR_INDEX = 0
masker = Masker()
openpose = OpenPose(0)
DRESS = 7 # COCO dataset format


def create_mask_v2_for_dresscode(base_path: Text, category: Text, overwrite=False):
    NUMBER_OF_ERRORS = 0
    ERROR_FILENAME = []
    target_folder = 'mask_v2'
    os.makedirs(osp.join(base_path, target_folder), exist_ok=True)
    files_list = os.listdir(osp.join(base_path, 'images'))
    for file_name in tqdm(files_list):
        file_path = osp.join(base_path, 'images', file_name)
        indicator_idx = file_path.split('/')[-1].split('.')[0].split('_')[-1]
        if int(indicator_idx) == IMAGE_INDICATOR_INDEX:
            with Image.open(file_path) as image:
                save_path = osp.join(base_path, target_folder, file_name)
                if os.path.isfile(save_path) and not overwrite:
                    print(f'IGNORE: {save_path}')
                    continue
                try:
                    mask, body_parse = masker.create_mask(
                        image,
                        category=category,
                        return_img=True,
                        return_body_parse=True
                    )
                    if category == 'lower_body' and DRESS in np.array(body_parse):
                        continue
                    mask.save(save_path, quality=100, subsampling=0)
                except IndexError:
                    NUMBER_OF_ERRORS += 1
                    ERROR_FILENAME.append(file_path)
                    continue
    print(f'NUMBER OF ERRORS: {NUMBER_OF_ERRORS}')
    print(f'ERROR_LIST: {ERROR_FILENAME}')

    
def create_densepose_for_dresscode(base_path: Text):
    NUMBER_OF_ERRORS = 0
    ERROR_FILENAME = []
    target_folder = 'dense_modified'
    os.makedirs(osp.join(base_path, target_folder), exist_ok=True)
    files_list = os.listdir(osp.join(base_path, 'images'))
    for file_name in tqdm(files_list):
        file_path = osp.join(base_path, 'images', file_name)
        indicator_idx = file_path.split('/')[-1].split('.')[0].split('_')[-1]
        if int(indicator_idx) == IMAGE_INDICATOR_INDEX:
            with Image.open(file_path) as image:
                save_path = osp.join(base_path, target_folder, file_name)
                if os.path.isfile(save_path):
                    print(f'IGNORE: {save_path}')
                    continue
                try:
                    dense = get_densepose_map(file_path, size=image.size)
                    dense.save(save_path)
                except IndexError:
                    NUMBER_OF_ERRORS += 1
                    ERROR_FILENAME.append(file_path)
                    continue
    print(f'NUMBER OF ERRORS: {NUMBER_OF_ERRORS}')
    print(f'ERROR_LIST: {ERROR_FILENAME}')


def create_skeleton_for_dresscode(base_path: Text, overwrite=False):
    NUMBER_OF_ERRORS = 0
    ERROR_FILENAME = []
    target_folder = 'skeleton_modified'
    os.makedirs(osp.join(base_path, target_folder), exist_ok=True)
    files_list = os.listdir(osp.join(base_path, 'images'))
    for file_name in tqdm(files_list):
        file_path = osp.join(base_path, 'images', file_name)
        indicator_idx = file_path.split('/')[-1].split('.')[0].split('_')[-1]
        if int(indicator_idx) == IMAGE_INDICATOR_INDEX:
            with Image.open(file_path) as image:
                save_path = osp.join(base_path, target_folder, file_name)
                if os.path.isfile(save_path) and not overwrite:
                    print(f'IGNORE: {save_path}')
                    continue
                try:
                    skeleton = openpose(image)[1]
                    skeleton = Image.fromarray(skeleton)
                    skeleton.save(save_path, quality=100, subsampling=0)
                except IndexError:
                    NUMBER_OF_ERRORS += 1
                    ERROR_FILENAME.append(file_path)
                    continue
    print(f'NUMBER OF ERRORS: {NUMBER_OF_ERRORS}')
    print(f'ERROR_LIST: {ERROR_FILENAME}')


def create_agnostic_from_mask_for_dresscode(base_path: Text):
    agnostic_folder = 'agnostic_v2'
    agnostic_path = osp.join(base_path, agnostic_folder)
    os.makedirs(agnostic_path, exist_ok=True)
    mask_path = osp.join(base_path, 'mask_v2')
    mask_names = os.listdir(mask_path)
    for mask_name in tqdm(mask_names):
        # person image
        tmp = None
        # number of images in the /images folder are always greater than ones in /mask_v2
        for image_name in os.listdir(osp.join(base_path, 'images')):
            if osp.splitext(mask_name)[0] == osp.splitext(image_name)[0]:
                tmp = image_name
                break
        assert tmp is not None
        img = osp.join(base_path, 'images', tmp)
        img = Image.open(img)
        img = np.array(img)
        # mask image
        mask = osp.join(mask_path, mask_name)
        mask = Image.open(mask)
        mask = mask.convert('L') # convert to Grayscale
        mask = mask.point(lambda i: 255 if i > 127 else 0) # thresholding -> np.unique(...) = [0, 255]
        mask = mask.convert('1') # convert to Bilevel (binary) iamge
        mask = np.array(mask, dtype=np.int8) # [0, 1] numpy array (single channel)
        mask = np.stack([mask] * 3)
        mask = rearrange(mask, 'c h w -> h w c')
        agnostic = np.where(mask, np.ones_like(img) * 127, img)
        agnostic = Image.fromarray(agnostic)
        agnostic.save(osp.join(agnostic_path, mask_name), quality=100, subsampling=0)        


def create_mask_v2_for_dresscode_parallel(base_path, category):
    """ This function doesn't work.
    """
    @delayed
    @wrap_non_picklable_objects
    def _create_mask_and_save(img: Image.Image, save_path: Text, category: Text):
        mask = masker.create_mask(img, category, return_img=True)
        mask.save(save_path)
        return mask

    os.makedirs(osp.join(base_path, 'mask_v2'), exist_ok=True)
    filenames = os.listdir(osp.join(base_path, 'images'))
    imgs = []
    save_paths = []
    categories = [category] * len(filenames)
    for file_name in tqdm(filenames):
        file_path = osp.join(base_path, 'images', file_name)
        indicator_idx = file_path.split('/')[-1].split('.')[0].split('_')[-1]
        if int(indicator_idx) == IMAGE_INDICATOR_INDEX:
            with Image.open(file_path) as img:
                imgs.append(img)
            save_path = osp.join(base_path, 'mask_v2', file_name)
            save_paths.append(save_path)

    Parallel(n_jobs=4, backend='multiprocessing')(_create_mask_and_save(img, save_path, category) for img, save_path, category in zip(imgs, save_paths, categories))
            

if __name__ == '__main__':
    fire.Fire()