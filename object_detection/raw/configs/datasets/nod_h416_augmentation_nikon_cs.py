
mean = (800.96014, 895.1913,  896.5624,  765.8472)

load_pipeline = [
    dict(type='LoadNumpyFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=(640, 416),
        recompute_bbox=True,
        allow_negative_crop=False),
    dict(
        type='RandomResize', scale=[(640, 320), (640, 416)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 416), pad_val=dict(img=mean)),

]

train_pipeline = [
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', "valid_img_shape"))
]

test_pipeline = [
    dict(type='LoadNumpyFromFile'),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 416), keep_ratio=True),

    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

