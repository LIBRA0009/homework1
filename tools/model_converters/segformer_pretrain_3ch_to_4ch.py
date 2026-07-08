# Copyright (c) OpenMMLab. All rights reserved.
"""Convert a 3-channel SegFormer/MiT checkpoint to 4-channel RGBD input.

The first patch-embedding convolution is expanded from RGB to RGBD. The depth
channel is initialized as the mean of the RGB kernels.
"""

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Expand SegFormer pretrain checkpoint from 3 to 4 channels')
    parser.add_argument('src', help='Source 3-channel checkpoint')
    parser.add_argument('dst', help='Destination 4-channel checkpoint')
    return parser.parse_args()


def find_first_patch_embed_key(state_dict):
    preferred = [
        'layers.0.0.projection.weight',
        'backbone.layers.0.0.projection.weight',
    ]
    for key in preferred:
        if key in state_dict and state_dict[key].ndim == 4:
            return key
    for key, value in state_dict.items():
        if key.endswith('projection.weight') and value.ndim == 4 \
                and value.shape[1] == 3:
            return key
    raise KeyError('Cannot find the first 3-channel patch embedding weight.')


def main():
    args = parse_args()
    checkpoint = torch.load(args.src, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    key = find_first_patch_embed_key(state_dict)
    weight = state_dict[key]
    if weight.shape[1] != 3:
        raise ValueError(f'{key} is expected to have 3 input channels, '
                         f'but got shape {tuple(weight.shape)}')

    depth_weight = weight.mean(dim=1, keepdim=True)
    state_dict[key] = torch.cat([weight, depth_weight], dim=1)
    torch.save(checkpoint, args.dst)
    print(f'Converted {key}: {tuple(weight.shape)} -> '
          f'{tuple(state_dict[key].shape)}')
    print(f'Saved to {args.dst}')


if __name__ == '__main__':
    main()
