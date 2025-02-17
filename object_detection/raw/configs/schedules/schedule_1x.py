# training schedule for 4x
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1),
    # dict(
    #     type='CosineAnnealingLR',
    #     begin=0,
    #     T_max=max_epochs,
    #     end=max_epochs,
    #     by_epoch=True,
    #     eta_min=0)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001, nesterov=True
        ),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
