import random
from typing import List, Tuple
from os import path as osp
import os
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from cleanfid import fid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from prettytable import PrettyTable
from src.models.utils import load_model
from src.pipelines.spacat_pipeline import TryOnPipeline
from src.dataset.vitonhd import VITONHDDataset
from src.dataset.dresscode import DressCodeDataset
from src.utils import make_custom_stats, get_project_root


PROJECT_ROOT_PATH = get_project_root()


def parse_args():
    parser = argparse.ArgumentParser(description='Script to eval trained models.')
    parser.add_argument(
        '--model_path',
        type=lambda x: x.split(" "),
        help='Path to model folder.'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--vitonhd_datapath',
        type=str,
    )
    parser.add_argument(
        '--dresscode_datapath',
        type=str
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory to save generated images.'
    )
    parser.add_argument(
        '--order',
        type=str,
        default='paired',
        help='Pair or Unpaired setting ("paired" | "unpaired")'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Height of input image of UNet.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=384,
        help='Height of input image of UNet.'
    )
    parser.add_argument(
        '--mask_strategy',
        type=str,
        default='dilated_relaxed',
        help='Input mask type for try-on region.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Whether or not to evaluate model on the given dataset.'
    )
    parser.add_argument(
        '--save_metrics_to_file',
        action='store_true'
    )

    return parser.parse_args()


class GroundTruthDataLoader(Dataset):
    def __init__(self, dataset_path: str, dataset_name: str, transform: transforms, size: Tuple[int]):
        self.dataroot = dataset_path
        self.dataset = dataset_name
        self.transform = transform
        self.size = size

        CATEGORIES = ['lower_body', 'upper_body', 'dresses']
        if dataset_name == 'dresscode':
            dresscode_filesplit = os.path.join(dataset_path, f"test_pairs_paired.txt")
            with open(dresscode_filesplit, 'r') as f:
                lines = f.read().splitlines()
            self.paths = sorted([
                osp.join(dataset_path, category, 'images', line.strip().split()[0]) for line in lines for
                category in CATEGORIES if
                osp.exists(osp.join(dataset_path, category, 'images', line.strip().split()[0]))
            ])
        elif dataset_name == 'vitonhd':
            self.paths = sorted([osp.abspath(entry) for entry in os.scandir(osp.join(dataset_path, 'test', 'image'))])
        else:
            raise ValueError(f'{dataset_name} is not supported.')

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_name = osp.basename(osp.splitext(self.paths[idx])[0])
        img = Image.open(self.paths[idx]).resize(self.size)
        img_tensor = self.transform(img)
        item = {'image': img_tensor, 'name': img_name}
        return item


class PredictionDataLoader(Dataset):
    def __init__(self, datapath: str, transform: transforms, size: Tuple[int]):
        self.datapath = datapath
        self.transform = transform
        self.size = size
        self.paths = sorted([osp.abspath(entry) for entry in os.scandir(datapath)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = osp.basename(osp.splitext(self.paths[idx])[0])
        img = Image.open(self.paths[idx]).resize(self.size)
        img_tensor = self.transform(img)
        item = {'image': img_tensor, 'name': img_name}
        return item


if __name__ == '__main__':
    args = parse_args()

    assert isinstance(args.model_path, List)

    table = PrettyTable()
    fields = ['Model', 'FID']
    row = []
    
    # *model_path* path argument should follow format "ROOT/../MODEL_NAME/CKPT_NAME"
    for model_path in args.model_path:
        unet, vae, scheduler = load_model(model_path, dtype=torch.float16)
        pipeline = TryOnPipeline(unet=unet, vae=vae, scheduler=scheduler).to(args.device)
        if args.dataset_name == 'vitonhd':
            test_set = VITONHDDataset(
                args.vitonhd_datapath,
                use_trainset=False,
                height=args.height,
                width=args.width,
                use_dilated_relaxed_mask=True
            )
        elif args.dataset_name == 'dresscode':
            test_set = DressCodeDataset(
                args.dresscode_datapath,
                phase='test',
                order=args.order,
                h=args.height,
                w=args.width,
                use_dilated_relaxed_mask=True
            )
        else:
            raise ValueError('Unsupported dataset. Allowed Datasets are ("vitonhd", "dresscode").')
        
        test_dataloader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        ckpt_name = model_path.split('/')[-1]
        save_dir = osp.join(args.output_dir, f'{args.dataset_name}_{args.order}', f'{ckpt_name}')
        if not osp.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=False)

        num_images = len(os.listdir(save_dir))
        if num_images == len(test_set):
            # Check whether or not to generate try-on images.
            # This should be an effective way to not wasting time & computational cost.
            print(f'\nNo generation process is done because there is already generated images under the folder {save_dir}')
        else:
            # Generate try-on images
            print(f'\nCKPT: {ckpt_name}')
            print(f'Generating try-on images on {args.dataset_name} ({args.order} setting) ...\n')
            for idx, batch in enumerate(tqdm(test_dataloader)):
                with torch.inference_mode():
                    with torch.amp.autocast(args.device):
                        images = pipeline(
                            image=batch['image'].to(args.device),
                            mask_image=batch['mask'].to(args.device),
                            densepose_image=batch['densepose'].to(args.device),
                            cloth_image=batch['cloth_raw'].to(args.device),
                            height=args.height,
                            width=args.width,
                            guidance_scale=1.5
                        ).images
                        for img, name in zip(images, batch['im_name']):
                            img.save(osp.join(save_dir, f'{name}'), quality=100, subsampling=0)
            print('\nGeneration Done.\n')

        # EVALUATION
        if args.eval:
            if args.dataset_name == 'vitonhd':
                dataset_path = args.vitonhd_datapath
            elif args.dataset_name == 'dresscode':
                dataset_path = args.dresscode_datapath
            else:
                raise ValueError('Supported Datasets: VITON-HD, DressCode')
            
            # FID
            print(f'Compute FID score\n')
            if not fid.test_stats_exists(name=args.dataset_name, mode='clean'):
                # makes dataset statistics (features from InceptionNet-v3 by default)
                make_custom_stats(dataset_name=args.dataset_name, dataset_path=dataset_path)
            fid_score = fid.compute_fid(
                fdir1=save_dir,
                dataset_name=args.dataset_name,
                mode='clean',
                dataset_split='custom',
                verbose=True,
                use_dataparallel=False
            )
            fid_score = round(fid_score, 3)
            row += [fid_score]
            
            if args.order == 'paired':
                # SSIM, LPIPS
                print('\nCompute SSIM & LPIPS\n')
                fields += ['SSIM', 'LPIPS']
                transform = transforms.ToTensor()
                pred_dataset = PredictionDataLoader(
                    datapath=save_dir,
                    transform=transform,
                    size=(args.width, args.height)
                )
                gt_dataset = GroundTruthDataLoader(
                    dataset_path=dataset_path,
                    dataset_name=args.dataset_name,
                    transform=transform,
                    size=(args.width, args.height)
                )
                pred_dataloader = DataLoader(
                    pred_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers
                )
                gt_dataloader = DataLoader(
                    gt_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers
                )

                ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(args.device)
                lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(args.device)

                for gt_batch, pred_batch in tqdm(zip(gt_dataloader, pred_dataloader)):
                    gt_images, gt_names = gt_batch['image'], gt_batch['name']
                    pred_images, pred_names = pred_batch['image'], pred_batch['name']
                    # assert pred_names == gt_names, 'Predicted images and ground truth ones are not matching.'
                    pred_images = pred_images.to(args.device)
                    gt_images = gt_images.to(args.device)
                    ssim.update(pred_images, gt_images)
                    lpips.update(pred_images, gt_images)
                ssim_score = round(ssim.compute().item(), 3)
                lpips_score = round(lpips.compute().item(), 3)
                row += [ssim_score, lpips_score]

            table.field_names = fields
            row.insert(0, ckpt_name)
            table.add_row(row)
            print(f'\n{table}')

            if args.save_metrics_to_file:
                model_name = model_path.split('/')[-2]
                save_dir = osp.join(PROJECT_ROOT_PATH, 'tmp', 'metrics', args.dataset_name, args.order)
                os.makedirs(save_dir, exist_ok=True)
                file = osp.join(save_dir, f'{model_name}.txt')
                with open(file, 'w') as f:
                    f.write(table.get_string())
                print(f'Saved metrics to {file}\n')

        del unet
        del vae
        del scheduler
        del pipeline
        torch.cuda.empty_cache()