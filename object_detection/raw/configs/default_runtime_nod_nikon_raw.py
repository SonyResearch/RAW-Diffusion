_base_ = [
    'default_runtime.py'
]

default_hooks = dict(
    visualization=dict(type='RawDetVisualizationHook', draw=True, vis_preprocessor_args=dict(type="DataToRGB", value_min=600, value_max=16383, raw_gamma=1 / 5)),
)