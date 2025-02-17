_base_ = [
    "nod_h416_augmentation_rgb.py",
]

# dataset settings
dataset_type = 'NODDataset'
data_root = '../data/NOD_processed/NOD_h416_d32/'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type="SampleConcatDataset",
        n=2571,
        f=1.0,
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/rawpy_new_Sony_RX100m7_train_100_0.json',
                data_prefix=dict(img='rawpy_new_Sony_RX100m7_train/'),
                pipeline=[*_base_.load_pipeline, *_base_.train_pipeline,],
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
        ann_file='annotations/rawpy_new_Sony_RX100m7_test.json',
        data_prefix=dict(img='rawpy_new_Sony_RX100m7_test/'),
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
        ann_file='annotations/rawpy_new_Sony_RX100m7_test.json',
        data_prefix=dict(img='rawpy_new_Sony_RX100m7_test/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/rawpy_new_Sony_RX100m7_test.json',
    metric='bbox',
    format_only=False)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/rawpy_new_Sony_RX100m7_test.json',
    metric='bbox',
    format_only=False)

custom_hooks = [
    dict(type='TorchinfoHook', input_size=(3, 640, 416))
]
