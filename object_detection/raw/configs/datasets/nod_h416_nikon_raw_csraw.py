import os

_base_ = [
    "nod_h416_augmentation_nikon_cs.py",
]

# dataset settings
dataset_type = 'NODDataset'
data_root = '../data/NOD_processed/NOD_h416_d32/'

raw_diffusion_path = "../experiments/NOD_h416_d32_Nikon750_train_100_0_R3206_256_4_70k_rawdiffusion_1000_linear_model_RAWDiffusionModel_32_64_2_8_A16-8_EDSR_4_64_False_midatt_l21.0_l11.0_logl11.0_linear_bl_tanh_0/"
cs_raw = os.path.join(raw_diffusion_path, "inference_sampling/Cityscapes_h416_train_ddim24")

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type="SampleConcatDataset",
        n=3206,
        f=0.05,
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/raw_new_Nikon750_train_100_0.json',
                data_prefix=dict(img='raw_new_Nikon750_train/'),
                pipeline=[*_base_.load_pipeline, *_base_.train_pipeline,],
            ),
            dict(
                type=dataset_type,
                data_root=cs_raw,
                ann_root="../data/Cityscapes_processed/Cityscapes_h416",
                ann_file='annotations/train.json',
                data_prefix=dict(img='leftImg8bit/'),
                pipeline=[*_base_.load_pipeline, *_base_.train_pipeline,],
                replace_filename={".png": "_pred_u16.hdf5"},
                normalization=((0, 65535), (600, 16383))
            )
        ]
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/raw_new_Nikon750_test.json',
        data_prefix=dict(img='raw_new_Nikon750_test/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/raw_new_Nikon750_test.json',
        data_prefix=dict(img='raw_new_Nikon750_test/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/raw_new_Nikon750_test.json',
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/raw_new_Nikon750_test.json',
    metric='bbox',
    format_only=False)

custom_hooks = [
    dict(type='TorchinfoHook', input_size=(4, 640, 416))
]
