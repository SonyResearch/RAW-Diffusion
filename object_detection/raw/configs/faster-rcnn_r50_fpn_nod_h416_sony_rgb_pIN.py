_base_ = [
    './datasets/nod_h416_sony_rgb.py',
    "./models/faster-rcnn_r50_fpn.py",
    'schedules/schedule_4x.py',  # 'mmdet::_base_/schedules/schedule_2x.py', 
    'default_runtime.py',
]

normalization = "meanstd" # None, meanstd, max

model = dict(
     data_preprocessor=dict(
        _delete_=True,
        type='MultichannelDetDataPreprocessor',
        mean=[123.675, 116.28 , 103.53], # (0.485, 0.456, 0.406) * 255,
        std=[58.395, 57.12 , 57.375], # (0.229, 0.224, 0.225) * 255,
        normalize_max=False,
        pad_size_divisor=1),
    backbone=dict(
        in_channels=3,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # init_cfg=dict(_delete_=True),
    ),
)
