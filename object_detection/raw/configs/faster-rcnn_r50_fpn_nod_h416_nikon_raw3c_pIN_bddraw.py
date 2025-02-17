_base_ = [
    './datasets/nod_h416_nikon_raw_bddraw.py',
    "./models/faster-rcnn_r50_fpn.py",
    'schedules/schedule_4x.py',  # 'mmdet::_base_/schedules/schedule_2x.py', 
    'default_runtime_nod_nikon_raw.py',
]

normalization = "meanstd" # None, meanstd, max

if normalization is None:
    mean, std = None, None
elif normalization == "max":
    mean = (0, 0, 0, 0)
    std = (16383, 16383, 16383, 16383)
elif normalization == "meanstd":
    mean = (800.96014, 895.1913,  896.5624,  765.8472)
    std = (782.79034, 963.45276, 963.0715,  752.7042)

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