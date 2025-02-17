_base_ = [
    './datasets/nod_h416_sony_raw_csraw.py',
    "./models/faster-rcnn_r50_fpn.py",
    'schedules/schedule_4x.py',  # 'mmdet::_base_/schedules/schedule_2x.py', 
    'default_runtime_nod_sony_raw.py',
]

mean = (1037.3071,  1202.6705,  1202.6272, 963.06726)
std = (884.8285, 1169.6968, 1170.2325,  763.7829)

model = dict(
     data_preprocessor=dict(
        _delete_=True,
        type='MultichannelDetDataPreprocessor',
        mean=mean,
        std=std,
        normalize_max=False,
        rggb_to_rgb=True,
        pad_size_divisor=1
    ),
    backbone=dict(
        in_channels=3,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # init_cfg=dict(_delete_=True),
    ),
)