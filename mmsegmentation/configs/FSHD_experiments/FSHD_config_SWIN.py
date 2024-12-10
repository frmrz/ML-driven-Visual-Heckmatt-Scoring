default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    dataset_name='FSHD_KNET_SWIN',
    vis_backends=vis_backends,
    save_dir='/media/francesco/DEV001/PROJECT-THYROID/RESULTS/FSHD',
    classes=[
        'background',
        'Biceps_brachii', # 001 - 1 for label
        'Deltoideus', # 002
        'Depressor_anguli_oris', # 003
        'Digastricus', # 004
        'Gastrocnemius_medial_head', # 008
        'Geniohyoideus', # 009
        'Masseter', # 011
        'Mentalis', # 012
        'Orbicularis_oris', # 013
        'Rectus_abdominis', # 015
        'Rectus_femoris', # 016
        'Temporalis', # 017
        'Tibialis_anterior', # 018
        'Trapezius', # 019
        'Vastus_lateralis', # 020
        'Zygomaticus'
    ],
    palette=[(0, 0, 0),       # black
        (128, 0, 128),   # purple
        (0, 128, 128),   # teal
        (128, 128, 128), # gray
        (255, 0, 0),     # red
        (0, 255, 0),     # lime
        (255, 255, 0),   # yellow
        (0, 0, 255),     # blue
        (255, 0, 255),   # fuchsia
        (0, 255, 255),   # aqua
        (192, 192, 192), # silver
        (255, 255, 255), # white
        (255, 99, 71),   # tomato
        (255, 69, 0),    # orange-red
        (255, 165, 0),   # orange
        (255, 215, 0),   # gold
        (46, 139, 87)],

    name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# dataset settings
dataset_type = 'FSHD'
data_root = 'data/FSHD_v3_f0'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/test',
            seg_map_path='ann_dir/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# model config
norm_cfg = dict(type='BN', requires_grad=True)
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'  # noqa

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    size=crop_size,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

num_stages = 3
conv_kernel_size = 1

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint_file,
    backbone=dict(
        # _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=num_stages,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=17,
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN'))) for _ in range(num_stages)
        ],
                kernel_generate_head=dict(
                    type='UPerHead',
                    in_channels=[192, 384, 768, 1536],
                    in_index=[0, 1, 2, 3],
                    pool_scales=(1, 2, 3, 6),
                    channels=512,
                    dropout_ratio=0.1,
                    num_classes=17,
                    norm_cfg=norm_cfg,
                    align_corners=False,
                    loss_decode=dict(
                        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                        class_weight=[0.69314718, 4.16186531, 4.7185577, 7.67713145, 6.1383027, 4.23960274,
         6.31456066, 4.84142826, 7.56522167, 8.08825252, 5.78304885, 5.15540083,
         4.53714328, 4.25948254, 5.17739035, 4.95696997, 6.84442111]))),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=17,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
            class_weight=[0.69314718, 4.16186531, 4.7185577, 7.67713145, 6.1383027, 4.23960274,
         6.31456066, 4.84142826, 7.56522167, 8.08825252, 5.78304885, 5.15540083,
         4.53714328, 4.25948254, 5.17739035, 4.95696997, 6.84442111])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='OptimWrapper',
    # modify learning rate following the official implementation of Swin Transformer # noqa
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=dict(max_norm=1, norm_type=2))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=1000,
        end=80000,
        milestones=[60000, 72000],
        by_epoch=False,
    )
]