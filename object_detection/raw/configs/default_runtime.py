_base_ = [
    'mmdet::_base_/default_runtime.py'
]
default_hooks = dict(
    visualization=dict(type='RawDetVisualizationHook', draw=True, vis_preprocessor_args=dict(type="DataToRGB", raw_gamma=0.3), interval_epoch=10),
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(max_keep_ckpts=2)
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="AimVisBackendv2",
        init_kwargs=dict(experiment="mmdetection"),
    ),
] 
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)
randomness=dict(seed=0)

custom_imports = dict(imports=[
    "raw.hooks",
    "raw.visualization.visualization_hook",
    "raw.data_preprocessors",
    "raw.datasets",
    "raw.datasets.nod",
    'raw.pipelines.loading',
    "raw.visualization",
    ], allow_failed_imports=False)

