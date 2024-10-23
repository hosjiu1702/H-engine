from typing import Text, Optional
from pathlib import Path
from shutil import unpack_archive
import os
import argparse
import requests
from tqdm import tqdm
from src.utils import get_project_root


PROJECT_ROOT_PATH = get_project_root()


def download_dataset(
    name: Text = 'vitonhd',
    save_path: Optional[Text] =  None,
):
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT_PATH, 'datasets', name)
        os.makedirs(save_path, exist_ok=True)

    if name == 'vitonhd':
        url = 'https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=1&dl=1'
        r = requests.get(url, stream=True)
        file_name = 'vitonhd'
        file_ext = '.zip'
        file_path = Path(save_path, file_name).with_suffix(file_ext)

        if file_path.exists:
            print(f'The {file_name} (zipped) dataset was downloaded before.')
            return file_path

        with open(file_path, 'wb') as f:
            # increase chunk size to could decrease download time
            for chunk in tqdm(r.iter_content(chunk_size=128)):
                f.write(chunk)

    print(f'Finish donwloading {name} dataset.')
    return file_path


def unzip(file_path: Text):
    extract_dir = os.path.dirname(file_path)
    print('Be extracting ...')
    unpack_archive(file_path, extract_dir)
    print('Unzipped done.')


if __name__ == '__main__':
    dataset_path = download_dataset(name='vitonhd')
    unzip(dataset_path)