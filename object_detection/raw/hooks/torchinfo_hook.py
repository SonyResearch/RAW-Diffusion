from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from mmdet.testing._utils import demo_mm_inputs

import torch.nn as nn

@HOOKS.register_module()
class TorchinfoHook(Hook):
    
    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size

    def before_run(self, runner: Runner):
        
        from torchinfo import summary
        model = runner.model
        bs = 4
        packed_inputs = demo_mm_inputs(
            bs,
            image_shapes=[self.input_size for _ in range(bs)])
        
        data = model.data_preprocessor(packed_inputs, False)

        summary(model, **data, depth=12)
