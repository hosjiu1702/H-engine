from typing import List
from os import path as osp
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from cleanfid import fid
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


if __name__ == '__main__':
    args = parse_args()
    
    table = PrettyTable()
    table.field_names = ['Model', 'FID']

    assert isinstance(args.model_path, List)

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
        os.makedirs(save_dir, exist_ok=True)
        print(f'\nCKPT: {ckpt_name}')
        print(f'Generating try-on images on {args.dataset_name} ({args.order} setting) ...\n')
        for batch in tqdm(test_dataloader):
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
                        img.save(osp.join(save_dir, f'{name}.png'))
        print('\nGeneration Done.\n')

        if args.eval:
            # FID
            if args.dataset_name == 'vitonhd':
                dataset_path = args.vitonhd_datapath
            elif args.dataset_name == 'dresscode':
                dataset_path = args.dresscode_datapath
            else:
                raise ValueError('Supported Datasets: VITON-HD, DressCode')
            
            # makes dataset statistics
            if not fid.test_stats_exists(name=args.dataset_name, mode='clean'):
                make_custom_stats(dataset_name=args.dataset_name, dataset_path=dataset_path)

            print(f'Compute FID score for [{ckpt_name}]\n')
            fid_score = fid.compute_fid(
                fdir1=save_dir,
                dataset_name=args.dataset_name,
                mode='clean',
                dataset_split='custom',
                verbose=True,
                use_dataparallel=False
            )

            table.add_row([ckpt_name, fid_score])

        del unet
        del vae
        del scheduler
        del pipeline
        torch.cuda.empty_cache()

    print(f'\n{table}')

    if args.save_metrics_to_file:
        model_name = model_path.split('/')[-2]
        save_dir = osp.join(PROJECT_ROOT_PATH, 'tmp', 'metrics', args.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        file = osp.join(save_dir, f'{model_name}.txt')
        with open(file, 'w') as f:
            f.write(table.get_string())
        print(f'Saved metrics to {file}\n')
