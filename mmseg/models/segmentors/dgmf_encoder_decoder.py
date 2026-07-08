# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptSampleList, SampleList, add_prefix
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class DGMFEncoderDecoder(EncoderDecoder):
    """Encoder-decoder segmentor for RGB-D DGMF experiments.

    The dataloader provides a 4-channel tensor made by RGB plus offline depth.
    This segmentor sends RGB channels to the SegFormer backbone and passes the
    depth channel to the decode head for depth-geometric fusion.
    """

    @staticmethod
    def _split_rgb_depth(inputs: Tensor):
        assert inputs.size(1) == 4, (
            'DGMFEncoderDecoder expects 4-channel inputs: RGB plus depth.')
        return inputs[:, :3, :, :], inputs[:, 3:4, :, :]

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        rgb_inputs, _ = self._split_rgb_depth(inputs)
        x = self.backbone(rgb_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        rgb_inputs, depth = self._split_rgb_depth(inputs)
        x = self.backbone(rgb_inputs)
        if self.with_neck:
            x = self.neck(x)
        seg_logits = self.decode_head.predict([x, depth], batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        rgb_inputs, depth = self._split_rgb_depth(inputs)
        x = self.backbone(rgb_inputs)
        if self.with_neck:
            x = self.neck(x)

        losses = dict()
        losses.update(self._decode_head_forward_train([x, depth],
                                                      data_samples))

        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(x, data_samples))

        return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        rgb_inputs, depth = self._split_rgb_depth(inputs)
        x = self.backbone(rgb_inputs)
        if self.with_neck:
            x = self.neck(x)
        return self.decode_head.forward([x, depth])
