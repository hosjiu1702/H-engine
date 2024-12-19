#!/bin/python
from typing import Text
import os
from os import path as osp
from PIL import Image
from tqdm import tqdm
import fire
# from joblib import Parallel, delayed, wrap_non_picklable_objects
from src.utils.mask_v2 import Maskerv2 as Masker


IMAGE_INDICATOR_INDEX = 0
masker = Masker()


def create_mask_v2_for_dresscode(base_path: Text, category: Text):
    NUMBER_OF_ERRORS = 0
    ERROR_FILENAME = []
    os.makedirs(osp.join(base_path, 'mask_v2'), exist_ok=True)
    files_list = os.listdir(osp.join(base_path, 'images'))
    for file_name in tqdm(files_list):
        file_path = osp.join(base_path, 'images', file_name)
        indicator_idx = file_path.split('/')[-1].split('.')[0].split('_')[-1]
        if int(indicator_idx) == IMAGE_INDICATOR_INDEX:
            with Image.open(file_path) as image:
                save_path = osp.join(base_path, 'mask_v2', file_name)
                if os.path.isfile(save_path):
                    print(f'IGNORE: {save_path}')
                    continue
                try:
                    mask = masker.create_mask(image, category=category, return_img=True)
                    mask.save(save_path)
                except IndexError:
                    NUMBER_OF_ERRORS += 1
                    ERROR_FILENAME.append(file_path)
                    continue
    print(f'NUMBER OF ERRORS: {NUMBER_OF_ERRORS}')
    print(f'ERROR_LIST: {ERROR_FILENAME}')


def create_mask_v2_for_dresscode_parallel(base_path, category):
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
    fire.Fire(create_mask_v2_for_dresscode)
    # fire.Fire(create_mask_v2_for_dresscode_parallel)