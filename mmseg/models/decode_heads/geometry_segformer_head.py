# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .segformer_head import SegformerHead


class LocalGeometryBlock(nn.Module):
    """Enhance features with local structure variation."""

    def __init__(self, channels, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.geometry_proj = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        geometry = torch.abs(x - self.avg_pool(x))
        geometry = self.geometry_proj(geometry)
        gate = self.gate(torch.cat([x, geometry], dim=1))
        return x + gate * geometry


@MODELS.register_module()
class GeometrySegformerHead(SegformerHead):
    """SegFormer head with local geometry enhancement.

    The module uses RGB backbone features only. For selected high-level stages,
    it estimates local geometric variation with ``abs(x - AvgPool3x3(x))`` and
    injects the enhanced structure feature through a lightweight gate.
    """

    def __init__(self, geometry_indices=(2, 3), **kwargs):
        super().__init__(**kwargs)
        self.geometry_indices = set(geometry_indices)

        self.geometry_blocks = nn.ModuleDict()
        for idx in self.geometry_indices:
            assert 0 <= idx < len(self.in_channels)
            self.geometry_blocks[str(idx)] = LocalGeometryBlock(
                self.in_channels[idx],
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = list(self._transform_inputs(inputs))

        for idx in self.geometry_indices:
            inputs[idx] = self.geometry_blocks[str(idx)](inputs[idx])

        outs = []
        for idx in range(len(inputs)):
            outs.append(
                resize(
                    input=self.convs[idx](inputs[idx]),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        return out
