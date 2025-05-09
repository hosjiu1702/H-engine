from typing import Text, Optional
from pathlib import Path
from shutil import unpack_archive
import os
from os.path import isfile
import sys
import requests
from tqdm import tqdm
from src.utils import get_project_root


PROJECT_ROOT_PATH = get_project_root()


def _download_dataset(
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

        if isfile(file_path) or isfile(os.path.join(save_path, 'train_pairs.txt')):
            return None

        with open(file_path, 'wb') as f:
            # increase chunk size could decrease download time
            for chunk in tqdm(r.iter_content(chunk_size=128)):
                f.write(chunk)

    print(f'Finish donwloading {name} dataset.')
    return file_path


def _unzip(file_path: Text):
    extract_dir = os.path.dirname(file_path)
    print('Be extracting ...')
    unpack_archive(file_path, extract_dir)
    print('Unzipped done.')


def download(dataset_name: Text = 'vitonhd'):
    zipfile = _download_dataset(dataset_name)
    if zipfile is None:
        print('Maybe, your dataset was downloaded and extracted. Check it out!')
        return
    else:
        _unzip(zipfile)
        os.remove(zipfile)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    download('vitonhd')