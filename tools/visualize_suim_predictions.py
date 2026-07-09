# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import random
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SUIM segmentation predictions.')
    parser.add_argument('config', help='Config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file path.')
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='val',
        help='Dataset split to visualize.')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/suim_vis',
        help='Directory to save visualization images.')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=24,
        help='Number of samples to visualize.')
    parser.add_argument(
        '--indices',
        type=str,
        default=None,
        help='Comma-separated dataset indices. Overrides random sampling.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.55,
        help='Mask overlay opacity.')
    parser.add_argument(
        '--show-depth',
        action='store_true',
        help='Add depth map panel when depth npy files are available.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. key=value.')
    return parser.parse_args()


def build_runner(config, checkpoint, device, cfg_options=None):
    cfg = Config.fromfile(config)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    cfg.load_from = checkpoint
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])

    runner = Runner.from_cfg(cfg)
    load_checkpoint(runner.model, checkpoint, map_location='cpu')
    runner.model.to(device)
    runner.model.eval()
    return runner


def get_dataloader_cfg(cfg, split):
    if split == 'train':
        return cfg.train_dataloader
    if split == 'val':
        return cfg.val_dataloader
    return cfg.test_dataloader


def choose_indices(dataset_len, args):
    if args.indices:
        indices = [int(item) for item in args.indices.split(',') if item.strip()]
        return [idx for idx in indices if 0 <= idx < dataset_len]

    rng = random.Random(args.seed)
    indices = list(range(dataset_len))
    rng.shuffle(indices)
    return indices[:min(args.num_samples, dataset_len)]


def to_numpy_label(label):
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    label = np.asarray(label)
    if label.ndim == 3:
        label = label[0]
    return label.astype(np.int64)


def colorize_label(label, palette, ignore_index=255):
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls_id, cls_color in enumerate(palette):
        color[label == cls_id] = np.asarray(cls_color, dtype=np.uint8)
    color[label == ignore_index] = np.array([30, 30, 30], dtype=np.uint8)
    return color


def overlay_label(image, label, palette, opacity, ignore_index=255):
    color = colorize_label(label, palette, ignore_index)
    valid = label != ignore_index
    out = image.copy()
    blended = (image * (1.0 - opacity) + color * opacity).astype(np.uint8)
    out[valid] = blended[valid]
    return out


def make_error_map(pred, gt, ignore_index=255):
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != ignore_index
    correct = (pred == gt) & valid
    wrong = (pred != gt) & valid
    error[correct] = np.array([40, 180, 80], dtype=np.uint8)
    error[wrong] = np.array([220, 50, 50], dtype=np.uint8)
    error[~valid] = np.array([30, 30, 30], dtype=np.uint8)
    return error


def add_title(image, title):
    title_h = 32
    canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:] = image
    cv2.putText(
        canvas,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (20, 20, 20),
        2,
        cv2.LINE_AA)
    return canvas


def resize_to(image, size):
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


def find_depth_path(img_path):
    path = Path(img_path)
    parts = list(path.parts)
    if 'images' not in parts:
        return None
    image_pos = parts.index('images')
    if image_pos + 1 >= len(parts):
        return None
    split = parts[image_pos + 1]
    root = Path(*parts[:image_pos])
    depth_path = root / 'depth' / split / f'{path.stem}.npy'
    return depth_path if depth_path.exists() else None


def find_ann_path(img_path):
    path = Path(img_path)
    parts = list(path.parts)
    if 'images' not in parts:
        return None
    image_pos = parts.index('images')
    if image_pos + 1 >= len(parts):
        return None
    split = parts[image_pos + 1]
    root = Path(*parts[:image_pos])
    ann_path = root / 'annotations' / split / f'{path.stem}.png'
    return ann_path if ann_path.exists() else None


def render_depth(img_path, size):
    depth_path = find_depth_path(img_path)
    if depth_path is None:
        return None
    depth = np.load(depth_path).astype(np.float32)
    depth = np.nan_to_num(depth)
    depth = np.clip(depth, 0.0, 1.0)
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
    depth_u8 = (depth * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)


def get_img_path(data_sample):
    metainfo = data_sample.metainfo
    return metainfo.get('img_path') or metainfo.get('filename')


def load_gt_label(img_path, data_sample, size):
    ann_path = find_ann_path(img_path)
    if ann_path is not None:
        gt = mmcv.imread(str(ann_path), flag='unchanged')
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        return resize_to(gt.astype(np.uint8), size).astype(np.int64)

    if hasattr(data_sample, 'gt_sem_seg'):
        gt = to_numpy_label(data_sample.gt_sem_seg.data)
        return resize_to(gt.astype(np.uint8), size).astype(np.int64)

    width, height = size
    return np.full((height, width), 255, dtype=np.int64)


def visualize_one(model, data, out_dir, palette, classes, args, sample_idx):
    with torch.no_grad():
        results = model.test_step(data)

    inputs = data['inputs']
    data_samples = data['data_samples']
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    for batch_idx, result in enumerate(results):
        data_sample = data_samples[batch_idx]
        img_path = get_img_path(data_sample)
        image = mmcv.imread(img_path, channel_order='rgb')
        height, width = image.shape[:2]

        pred = to_numpy_label(result.pred_sem_seg.data)
        pred = resize_to(pred.astype(np.uint8), (width, height)).astype(np.int64)

        gt = load_gt_label(img_path, data_sample, (width, height))

        gt_overlay = overlay_label(image, gt, palette, args.opacity)
        pred_overlay = overlay_label(image, pred, palette, args.opacity)
        error_map = make_error_map(pred, gt)

        panels = [
            add_title(image, 'Image'),
            add_title(gt_overlay, 'GT'),
            add_title(pred_overlay, 'Prediction'),
            add_title(error_map, 'Error: green=correct red=wrong')
        ]
        if args.show_depth:
            depth_panel = render_depth(img_path, (width, height))
            if depth_panel is not None:
                panels.insert(1, add_title(depth_panel, 'Depth'))

        montage = np.concatenate(panels, axis=1)
        stem = Path(img_path).stem
        out_file = osp.join(out_dir, f'{sample_idx:04d}_{stem}.png')
        mmcv.imwrite(cv2.cvtColor(montage, cv2.COLOR_RGB2BGR), out_file)


def main():
    args = parse_args()
    mkdir_or_exist(args.out_dir)

    runner = build_runner(args.config, args.checkpoint, args.device,
                          args.cfg_options)
    dataloader_cfg = get_dataloader_cfg(runner.cfg, args.split)
    dataloader = runner.build_dataloader(dataloader_cfg)
    dataset = dataloader.dataset
    indices = choose_indices(len(dataset), args)

    dataset_meta = getattr(runner.model, 'dataset_meta', None)
    if dataset_meta is None:
        dataset_meta = dataset.metainfo
    palette = dataset_meta['palette']
    classes = dataset_meta['classes']

    for order, dataset_idx in enumerate(indices):
        data = dataset[dataset_idx]
        data = dataloader.collate_fn([data])
        visualize_one(runner.model, data, args.out_dir, palette, classes, args,
                      dataset_idx)
        print(f'[{order + 1}/{len(indices)}] saved sample {dataset_idx}')

    print(f'Done. Visualizations are saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
