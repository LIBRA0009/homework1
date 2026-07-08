# Copyright (c) OpenMMLab. All rights reserved.
"""Generate offline Depth Anything V2 depth priors for SUIM.

Expected input layout:

    data/SUIM/images/train/*.png
    data/SUIM/images/val/*.png
    data/SUIM/images/test/*.png

Expected output layout:

    data/SUIM/depth/train/*.npy
    data/SUIM/depth/val/*.npy
    data/SUIM/depth/test/*.npy
    data/SUIM/depth_vis/*.jpg

The saved ``.npy`` files are float32 relative depth maps normalized per image
with P2/P98 clipping into [0, 1].
"""

import argparse
import json
import os
import os.path as osp
import random
import sys
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image


MODEL_CONFIGS = {
    'vits': {
        'encoder': 'vits',
        'features': 64,
        'out_channels': [48, 96, 192, 384],
    },
    'vitb': {
        'encoder': 'vitb',
        'features': 128,
        'out_channels': [96, 192, 384, 768],
    },
    'vitl': {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024],
    },
    'vitg': {
        'encoder': 'vitg',
        'features': 384,
        'out_channels': [1536, 1536, 1536, 1536],
    },
}
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Depth Anything V2 depth maps for SUIM')
    parser.add_argument(
        '--data-root',
        default='data/SUIM',
        help='Prepared SUIM root containing images/{train,val,test}.')
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/depth_anything_v2_vits.pth',
        help='Depth Anything V2 checkpoint path.')
    parser.add_argument(
        '--encoder',
        choices=tuple(MODEL_CONFIGS.keys()),
        default='vits',
        help='Depth Anything V2 encoder variant.')
    parser.add_argument(
        '--depth-anything-root',
        default='',
        help='Path to the cloned Depth-Anything-V2 repo. If omitted, the '
        'script expects depth_anything_v2 to be importable.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to process.')
    parser.add_argument(
        '--input-size',
        type=int,
        default=518,
        help='Depth Anything V2 inference input size.')
    parser.add_argument(
        '--device',
        default='cuda',
        help='Inference device, e.g. cuda, cuda:0, or cpu.')
    parser.add_argument(
        '--vis-count',
        type=int,
        default=50,
        help='Number of random visual quality-check images to save.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing .npy depth files.')
    return parser.parse_args()


def import_depth_anything(depth_anything_root: str):
    if depth_anything_root:
        sys.path.insert(0, osp.abspath(depth_anything_root))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Cannot import depth_anything_v2. Clone the official '
            'Depth-Anything-V2 repo and pass --depth-anything-root, for '
            'example: --depth-anything-root /path/to/Depth-Anything-V2') from exc
    return DepthAnythingV2


def list_images(path: str) -> List[str]:
    if not osp.isdir(path):
        return []
    return sorted(
        osp.join(path, name) for name in os.listdir(path)
        if osp.splitext(name)[1].lower() in IMAGE_EXTS)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    lo, hi = np.percentile(depth, [2, 98])
    if hi <= lo + 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    depth = (depth - lo) / (hi - lo)
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def depth_edges(depth: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(grad_x) + np.abs(grad_y)
    return normalize_depth(edge)


def to_gray_rgb(array: np.ndarray) -> Image.Image:
    gray = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(gray, mode='L').convert('RGB')


def save_visual_sample(img_path: str, depth: np.ndarray, out_path: str) -> None:
    image = Image.open(img_path).convert('RGB')
    depth_image = to_gray_rgb(depth).resize(image.size)
    edge_image = to_gray_rgb(depth_edges(depth)).resize(image.size)

    canvas = Image.new('RGB', (image.width * 3, image.height))
    canvas.paste(image, (0, 0))
    canvas.paste(depth_image, (image.width, 0))
    canvas.paste(edge_image, (image.width * 2, 0))
    canvas.save(out_path)


def load_model(args):
    DepthAnythingV2 = import_depth_anything(args.depth_anything_root)
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    return model


def generate_depth(model, img_path: str, input_size: int) -> np.ndarray:
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f'Failed to read image: {img_path}')
    depth = model.infer_image(image, input_size)
    return normalize_depth(depth)


def collect_visual_targets(split_to_images: Dict[str, List[str]], vis_count: int,
                           seed: int) -> set:
    all_items = []
    for split, paths in split_to_images.items():
        all_items.extend((split, path) for path in paths)
    rng = random.Random(seed)
    rng.shuffle(all_items)
    return set(all_items[:vis_count])


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA is not available. Falling back to CPU.')
        args.device = 'cpu'
    if not osp.isfile(args.checkpoint):
        raise FileNotFoundError(
            f'Checkpoint not found: {args.checkpoint}. Put your '
            'depth_anything_v2_vits.pth under checkpoints/ or pass '
            '--checkpoint explicitly.')

    split_to_images = {
        split: list_images(osp.join(args.data_root, 'images', split))
        for split in args.splits
    }
    vis_targets = collect_visual_targets(split_to_images, args.vis_count,
                                         args.seed)

    model = load_model(args)
    report = {
        'checkpoint': args.checkpoint,
        'encoder': args.encoder,
        'input_size': args.input_size,
        'splits': {},
    }

    vis_dir = osp.join(args.data_root, 'depth_vis')
    mkdir_or_exist(vis_dir)

    for split, img_paths in split_to_images.items():
        out_dir = osp.join(args.data_root, 'depth', split)
        mkdir_or_exist(out_dir)
        report['splits'][split] = {'images': len(img_paths), 'written': 0}
        print(f'Generating depth for {split}: {len(img_paths)} images')
        progress_bar = ProgressBar(len(img_paths))

        for img_path in img_paths:
            stem = osp.splitext(osp.basename(img_path))[0]
            out_path = osp.join(out_dir, stem + '.npy')
            if osp.exists(out_path) and not args.overwrite:
                progress_bar.update()
                continue

            depth = generate_depth(model, img_path, args.input_size)
            np.save(out_path, depth.astype(np.float32))
            report['splits'][split]['written'] += 1

            if (split, img_path) in vis_targets:
                save_visual_sample(
                    img_path, depth, osp.join(vis_dir, f'{split}_{stem}.jpg'))
            progress_bar.update()

    report_path = osp.join(args.data_root, 'depth_generation_report.json')
    with open(report_path, 'w') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
