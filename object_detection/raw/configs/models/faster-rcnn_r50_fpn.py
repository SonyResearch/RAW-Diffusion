_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py"
]

model = dict(
    data_preprocessor=dict(
        type='MultichannelDetDataPreprocessor',
        mean=None,
        std=None,
        normalize_max=False,
        pad_size_divisor=1),
    backbone=dict(
       type='ResNet',
        in_channels=3,
        norm_eval=True
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=3,
        )
    ),

    rpn_head=dict(
        anchor_generator=dict(
            scales=[4, 8],
        ),
    ),
)