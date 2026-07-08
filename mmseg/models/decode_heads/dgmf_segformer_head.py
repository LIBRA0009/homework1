# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .segformer_head import SegformerHead


class DGMFBlock(nn.Module):
    """Depth-geometric modulation fusion block."""

    def __init__(self, channels, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.depth_proj = ConvModule(
            1,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.geometry_proj = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.depth_edge_proj = ConvModule(
            1,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.Sigmoid())

        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _depth_edge(self, depth):
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    def forward(self, x, depth):
        depth = resize(
            input=depth,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False)
        depth_feat = self.depth_proj(depth)
        geometry_feat = self.geometry_proj(torch.abs(x - self.avg_pool(x)))
        depth_edge_feat = self.depth_edge_proj(self._depth_edge(depth))
        gate = self.gate(
            torch.cat([x, depth_feat, geometry_feat, depth_edge_feat], dim=1))

        return x + gate * depth_feat + (1.0 - gate) * geometry_feat


@MODELS.register_module()
class DGMFSegformerHead(SegformerHead):
    """SegFormer head with Depth-Geometric Modulation Fusion.

    Inputs are ``[features, depth]`` where features are RGB backbone outputs and
    depth is the fourth input channel after preprocessing.
    """

    def __init__(self, fusion_indices=(2, 3), **kwargs):
        super().__init__(**kwargs)
        self.fusion_indices = set(fusion_indices)

        self.dgmf_blocks = nn.ModuleDict()
        for idx in self.fusion_indices:
            assert 0 <= idx < len(self.in_channels)
            self.dgmf_blocks[str(idx)] = DGMFBlock(
                self.in_channels[idx],
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        features, depth = inputs
        features = list(self._transform_inputs(features))

        for idx in self.fusion_indices:
            features[idx] = self.dgmf_blocks[str(idx)](features[idx], depth)

        outs = []
        for idx in range(len(features)):
            outs.append(
                resize(
                    input=self.convs[idx](features[idx]),
                    size=features[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        return out
