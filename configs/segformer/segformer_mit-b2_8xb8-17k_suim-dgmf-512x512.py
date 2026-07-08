_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/suim_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    bgr_to_rgb=False,
    mean=[123.675, 116.28, 103.53, 127.5],
    std=[58.395, 57.12, 57.375, 127.5])

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

model = dict(
    type='DGMFEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=3,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(
        type='DGMFSegformerHead',
        num_classes=8,
        in_channels=[64, 128, 320, 512],
        fusion_indices=(2, 3)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadSUIMDepthAsChannel'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSUIMDepthAsChannel'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(pipeline=test_pipeline))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=17000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'dgmf_blocks': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=17000,
        by_epoch=False)
]

randomness = dict(seed=42)
