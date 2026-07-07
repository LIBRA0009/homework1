# Copyright (c) OpenMMLab. All rights reserved.
"""Prepare cleaned SUIM files for training.

Input is the output of ``tools/dataset_converters/suim.py``:

    img_dir/train_val/*.png
    ann_dir/train_val/*.png
    img_dir/test/*.png      optional
    ann_dir/test/*.png      optional

Output follows the experiment plan:

    images/train, images/val, images/test
    annotations/train, annotations/val, annotations/test
    splits/train.txt, splits/val.txt, splits/test.txt, splits/overfit8.txt
    cleaning_report.json
"""

import argparse
import json
import os
import os.path as osp
import random
import shutil
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image


SUIM_CLASSES = ('BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR')
SUIM_PALETTE = np.array(
    [[0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0],
     [255, 0, 255], [255, 255, 0], [255, 255, 255]],
    dtype=np.uint8)
VALID_LABELS = set(range(8)) | {255}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split and validate cleaned SUIM dataset')
    parser.add_argument('clean_dir', help='Cleaned SUIM folder')
    parser.add_argument('-o', '--out-dir', required=True, help='Output folder')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overfit-count', type=int, default=8)
    parser.add_argument('--vis-count', type=int, default=20)
    parser.add_argument(
        '--mode',
        choices=('copy', 'hardlink'),
        default='copy',
        help='How to place files in the output folder.')
    return parser.parse_args()


def list_png(path: str) -> List[str]:
    if not osp.isdir(path):
        return []
    return sorted(
        osp.join(path, name) for name in os.listdir(path)
        if osp.splitext(name)[1].lower() == '.png')


def stem(path: str) -> str:
    return osp.splitext(osp.basename(path))[0]


def pair_files(img_dir: str, ann_dir: str) -> Tuple[List[str], List[str], List[str]]:
    images = {stem(path): path for path in list_png(img_dir)}
    anns = {stem(path): path for path in list_png(ann_dir)}
    paired = sorted(set(images) & set(anns))
    missing_img = sorted(set(anns) - set(images))
    missing_ann = sorted(set(images) - set(anns))
    return paired, missing_img, missing_ann


def ensure_layout(out_dir: str) -> None:
    for rel_path in [
            'images/train', 'images/val', 'images/test', 'annotations/train',
            'annotations/val', 'annotations/test', 'splits', 'vis_samples'
    ]:
        mkdir_or_exist(osp.join(out_dir, rel_path))


def place_file(src: str, dst: str, mode: str) -> None:
    if mode == 'hardlink':
        if osp.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def write_split(path: str, stems: Iterable[str]) -> None:
    with open(path, 'w') as file:
        for item in stems:
            file.write(item + '\n')


def validate_and_count(img_path: str, ann_path: str) -> Dict:
    with Image.open(img_path) as image, Image.open(ann_path) as ann_image:
        if image.size != ann_image.size:
            return {
                'ok': False,
                'reason': 'size_mismatch',
                'image_size': list(image.size),
                'ann_size': list(ann_image.size),
            }
        ann = np.asarray(ann_image)

    if ann.ndim != 2:
        return {'ok': False, 'reason': 'annotation_not_single_channel'}

    labels, counts = np.unique(ann, return_counts=True)
    invalid = [int(label) for label in labels if int(label) not in VALID_LABELS]
    if invalid:
        return {'ok': False, 'reason': 'invalid_labels', 'labels': invalid}

    label_counts = {int(label): int(count) for label, count in zip(labels, counts)}
    present_classes = [int(label) for label in labels if 0 <= int(label) < 8]
    return {
        'ok': True,
        'label_counts': label_counts,
        'present_classes': present_classes,
    }


def colorize_label(label: np.ndarray) -> np.ndarray:
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, rgb in enumerate(SUIM_PALETTE):
        color[label == class_id] = rgb
    color[label == 255] = (128, 128, 128)
    return color


def save_visual_sample(img_path: str, ann_path: str, dst_path: str) -> None:
    image = Image.open(img_path).convert('RGB')
    label = np.asarray(Image.open(ann_path))
    label_color = Image.fromarray(colorize_label(label))
    overlay = Image.blend(image, label_color, alpha=0.45)

    canvas = Image.new('RGB', (image.width * 3, image.height))
    canvas.paste(image, (0, 0))
    canvas.paste(label_color, (image.width, 0))
    canvas.paste(overlay, (image.width * 2, 0))
    canvas.save(dst_path)


def prepare_split(args, split: str, stems: List[str], src_img_dir: str,
                  src_ann_dir: str) -> Dict:
    out_img_dir = osp.join(args.out_dir, 'images', split)
    out_ann_dir = osp.join(args.out_dir, 'annotations', split)
    report = {
        'count': 0,
        'invalid': [],
        'label_pixels': Counter(),
        'class_image_counts': Counter(),
    }

    print(f'Preparing {split}: {len(stems)} pairs')
    progress_bar = ProgressBar(len(stems))
    for item in stems:
        img_path = osp.join(src_img_dir, item + '.png')
        ann_path = osp.join(src_ann_dir, item + '.png')
        check = validate_and_count(img_path, ann_path)
        if not check['ok']:
            report['invalid'].append({'stem': item, **check})
            progress_bar.update()
            continue

        place_file(img_path, osp.join(out_img_dir, item + '.png'), args.mode)
        place_file(ann_path, osp.join(out_ann_dir, item + '.png'), args.mode)
        report['count'] += 1
        report['label_pixels'].update(check['label_counts'])
        report['class_image_counts'].update(check['present_classes'])
        progress_bar.update()

    return report


def main():
    args = parse_args()
    if not 0 < args.train_ratio < 1:
        raise ValueError('--train-ratio must be between 0 and 1')
    ensure_layout(args.out_dir)

    train_val_img_dir = osp.join(args.clean_dir, 'img_dir', 'train_val')
    train_val_ann_dir = osp.join(args.clean_dir, 'ann_dir', 'train_val')
    all_train_val, missing_img, missing_ann = pair_files(train_val_img_dir,
                                                         train_val_ann_dir)

    rng = random.Random(args.seed)
    shuffled = all_train_val[:]
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * args.train_ratio)
    train_stems = sorted(shuffled[:train_count])
    val_stems = sorted(shuffled[train_count:])
    overfit_stems = train_stems[:args.overfit_count]

    train_report = prepare_split(args, 'train', train_stems, train_val_img_dir,
                                 train_val_ann_dir)
    val_report = prepare_split(args, 'val', val_stems, train_val_img_dir,
                               train_val_ann_dir)

    test_img_dir = osp.join(args.clean_dir, 'img_dir', 'test')
    test_ann_dir = osp.join(args.clean_dir, 'ann_dir', 'test')
    test_stems, test_missing_img, test_missing_ann = pair_files(
        test_img_dir, test_ann_dir)
    test_report = prepare_split(args, 'test', test_stems, test_img_dir,
                                test_ann_dir) if test_stems else None

    write_split(osp.join(args.out_dir, 'splits', 'train.txt'), train_stems)
    write_split(osp.join(args.out_dir, 'splits', 'val.txt'), val_stems)
    write_split(osp.join(args.out_dir, 'splits', 'test.txt'), test_stems)
    write_split(osp.join(args.out_dir, 'splits', 'overfit8.txt'), overfit_stems)

    vis_stems = train_stems[:args.vis_count]
    for item in vis_stems:
        save_visual_sample(
            osp.join(args.out_dir, 'images', 'train', item + '.png'),
            osp.join(args.out_dir, 'annotations', 'train', item + '.png'),
            osp.join(args.out_dir, 'vis_samples', item + '.jpg'))

    report = {
        'classes': SUIM_CLASSES,
        'palette': SUIM_PALETTE.astype(int).tolist(),
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'source_train_val_pairs': len(all_train_val),
        'missing_image': missing_img,
        'missing_annotation': missing_ann,
        'test_missing_image': test_missing_img,
        'test_missing_annotation': test_missing_ann,
        'splits': {
            'train': train_report,
            'val': val_report,
            'test': test_report,
        },
        'overfit8': overfit_stems,
        'visual_samples': len(vis_stems),
    }

    def normalize(obj):
        if isinstance(obj, Counter):
            return {str(k): int(v) for k, v in sorted(obj.items())}
        raise TypeError(f'Unsupported type: {type(obj)}')

    with open(osp.join(args.out_dir, 'cleaning_report.json'), 'w') as file:
        json.dump(report, file, indent=2, default=normalize)
        file.write('\n')
    print(json.dumps(report, indent=2, default=normalize))


if __name__ == '__main__':
    main()
