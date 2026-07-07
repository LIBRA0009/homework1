# Copyright (c) OpenMMLab. All rights reserved.
"""Convert and clean SUIM annotations for MMSegmentation.

The converter is intentionally conservative. It does not modify the source
dataset, skips image/mask pairs with mismatched spatial sizes by default, and
maps polluted RGB mask pixels to the nearest SUIM palette color only when they
are within a configurable distance threshold.
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


SUIM_CLASSES = ('BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR')
SUIM_PALETTE = np.array(
    [
        (0, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
    ],
    dtype=np.int16)
TEST_MASK_DIR_TO_LABEL = {name: idx for idx, name in enumerate(SUIM_CLASSES)}
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
MASK_EXTS = ('.bmp', '.png', '.jpg', '.jpeg')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Clean and convert SUIM dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='SUIM root folder path')
    parser.add_argument(
        '-o',
        '--out-dir',
        required=True,
        help='Output folder for converted mmsegmentation dataset')
    parser.add_argument(
        '--max-color-distance',
        type=float,
        default=32.0,
        help='Maximum RGB Euclidean distance to snap a polluted mask pixel to '
        'a SUIM palette color. Farther pixels become ignore-index.')
    parser.add_argument(
        '--size-policy',
        choices=('skip', 'resize-mask'),
        default='skip',
        help='How to handle train_val image/mask size mismatches.')
    parser.add_argument(
        '--ignore-index',
        type=int,
        default=255,
        help='Label value for ignored pixels.')
    parser.add_argument(
        '--copy-test-images',
        action='store_true',
        help='Also copy TEST/images into img_dir/test.')
    parser.add_argument(
        '--convert-test-masks',
        action='store_true',
        help='Combine TEST/masks class-wise binary masks into ann_dir/test. '
        'The Saliency folder is ignored.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Scan and report problems without writing converted images/masks.')
    return parser.parse_args()


def iter_files(path: str, exts: Iterable[str]) -> List[str]:
    if not osp.isdir(path):
        return []
    exts = tuple(ext.lower() for ext in exts)
    return sorted(
        osp.join(path, name) for name in os.listdir(path)
        if osp.splitext(name)[1].lower() in exts)


def index_by_stem(paths: Iterable[str]) -> Dict[str, str]:
    return {osp.splitext(osp.basename(path))[0]: path for path in paths}


def ensure_dirs(out_dir: str, dry_run: bool) -> None:
    if dry_run:
        return
    for rel_path in [
            'img_dir/train_val',
            'ann_dir/train_val',
            'img_dir/test',
            'ann_dir/test',
    ]:
        mkdir_or_exist(osp.join(out_dir, rel_path))


def convert_color_mask(mask: np.ndarray, max_dist: float,
                       ignore_index: int) -> Tuple[np.ndarray, int, Counter]:
    flat = mask.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    diff = colors.astype(np.int32)[:, None, :] - SUIM_PALETTE[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest = np.argmin(dist2, axis=1).astype(np.uint8)
    min_dist2 = dist2[np.arange(dist2.shape[0]), nearest]
    unknown = min_dist2 > max_dist * max_dist

    color_labels = nearest
    color_labels[unknown] = ignore_index
    labels = color_labels[inverse].reshape(mask.shape[:2])

    exact_palette = min_dist2 == 0
    color_counts = np.bincount(inverse, minlength=len(colors))
    snapped = int(color_counts[~exact_palette & ~unknown].sum())
    unknown_count = int(color_counts[unknown].sum())

    label_counts = Counter(labels.reshape(-1).tolist())
    return labels.astype(np.uint8), snapped, unknown_count, label_counts


def clean_train_val(args) -> Dict:
    src_img_dir = osp.join(args.dataset_path, 'train_val', 'images')
    src_mask_dir = osp.join(args.dataset_path, 'train_val', 'masks')
    out_img_dir = osp.join(args.out_dir, 'img_dir', 'train_val')
    out_ann_dir = osp.join(args.out_dir, 'ann_dir', 'train_val')

    images = index_by_stem(iter_files(src_img_dir, IMAGE_EXTS))
    masks = index_by_stem(iter_files(src_mask_dir, MASK_EXTS))
    stems = sorted(set(images) | set(masks))

    report = {
        'split': 'train_val',
        'written': 0,
        'missing_image': [],
        'missing_mask': [],
        'size_mismatch': [],
        'snapped_pixels': 0,
        'ignored_pixels': 0,
        'label_counts': Counter(),
    }

    print(f'Converting train_val: {len(stems)} image/mask entries')
    progress_bar = ProgressBar(len(stems))
    for stem in stems:
        img_path = images.get(stem)
        mask_path = masks.get(stem)
        if img_path is None:
            report['missing_image'].append(stem)
            progress_bar.update()
            continue
        if mask_path is None:
            report['missing_mask'].append(stem)
            progress_bar.update()
            continue

        image = Image.open(img_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('RGB')
        try:
            if image.size != mask_image.size:
                mismatch = {
                    'stem': stem,
                    'image_size': list(image.size),
                    'mask_size': list(mask_image.size),
                }
                report['size_mismatch'].append(mismatch)
                if args.size_policy == 'skip':
                    progress_bar.update()
                    continue
                mask_image = mask_image.resize(image.size, RESAMPLE_NEAREST)

            labels, snapped, ignored, counts = convert_color_mask(
                np.asarray(mask_image), args.max_color_distance,
                args.ignore_index)
            report['snapped_pixels'] += snapped
            report['ignored_pixels'] += ignored
            report['label_counts'].update(counts)

            if not args.dry_run:
                image.save(osp.join(out_img_dir, stem + '.png'))
                Image.fromarray(labels).save(
                    osp.join(out_ann_dir, stem + '.png'))
            report['written'] += 1
            progress_bar.update()
        finally:
            image.close()
            mask_image.close()

    return report


def combine_binary_test_masks(mask_root: str, stem: str,
                              shape: Tuple[int, int]) -> Tuple[np.ndarray, int]:
    label = np.zeros(shape, dtype=np.uint8)
    overlaps = 0
    for class_name, class_id in TEST_MASK_DIR_TO_LABEL.items():
        if class_name == 'BW':
            continue
        class_dir = osp.join(mask_root, class_name)
        candidates = [
            osp.join(class_dir, stem + ext) for ext in MASK_EXTS
            if osp.exists(osp.join(class_dir, stem + ext))
        ]
        if not candidates:
            continue
        mask_image = Image.open(candidates[0]).convert('L')
        try:
            binary = np.asarray(mask_image) > 0
        finally:
            mask_image.close()
        if binary.shape != shape:
            binary = np.asarray(
                Image.fromarray(binary.astype(np.uint8) * 255).resize(
                    (shape[1], shape[0]), RESAMPLE_NEAREST)) > 0
        overlaps += int(np.count_nonzero((label != 0) & binary))
        label[binary] = class_id
    return label, overlaps


def convert_test(args) -> Optional[Dict]:
    if not args.copy_test_images and not args.convert_test_masks:
        return None

    src_img_dir = osp.join(args.dataset_path, 'TEST', 'images')
    src_mask_root = osp.join(args.dataset_path, 'TEST', 'masks')
    out_img_dir = osp.join(args.out_dir, 'img_dir', 'test')
    out_ann_dir = osp.join(args.out_dir, 'ann_dir', 'test')
    test_images = iter_files(src_img_dir, IMAGE_EXTS)

    report = {
        'split': 'test',
        'images_seen': len(test_images),
        'images_written': 0,
        'masks_written': 0,
        'overlap_pixels': 0,
    }

    print(f'Converting TEST: {len(test_images)} images')
    progress_bar = ProgressBar(len(test_images))
    for img_path in test_images:
        stem = osp.splitext(osp.basename(img_path))[0]
        image = Image.open(img_path).convert('RGB')
        try:
            width, height = image.size
            if args.copy_test_images and not args.dry_run:
                image.save(osp.join(out_img_dir, stem + '.png'))
            if args.copy_test_images:
                report['images_written'] += 1
        finally:
            image.close()

        if args.convert_test_masks:
            label, overlaps = combine_binary_test_masks(
                src_mask_root, stem, (height, width))
            report['overlap_pixels'] += overlaps
            if not args.dry_run:
                Image.fromarray(label).save(
                    osp.join(out_ann_dir, stem + '.png'))
            report['masks_written'] += 1
        progress_bar.update()

    return report


def dump_report(out_dir: str, report: Dict, dry_run: bool) -> None:
    def normalize(obj):
        if isinstance(obj, Counter):
            return {str(k): int(v) for k, v in sorted(obj.items())}
        raise TypeError(f'Unsupported type: {type(obj)}')

    text = json.dumps(report, indent=2, default=normalize)
    print(text)
    if not dry_run:
        with open(osp.join(out_dir, 'suim_clean_report.json'), 'w') as file:
            file.write(text + '\n')


def main():
    args = parse_args()
    if not 0 <= args.ignore_index <= 255:
        raise ValueError('--ignore-index must fit in an 8-bit mask')
    ensure_dirs(args.out_dir, args.dry_run)

    report = {
        'classes': SUIM_CLASSES,
        'palette': SUIM_PALETTE.astype(int).tolist(),
        'max_color_distance': args.max_color_distance,
        'size_policy': args.size_policy,
        'ignore_index': args.ignore_index,
        'train_val': clean_train_val(args),
    }
    test_report = convert_test(args)
    if test_report is not None:
        report['test'] = test_report

    dump_report(args.out_dir, report, args.dry_run)


if __name__ == '__main__':
    main()
