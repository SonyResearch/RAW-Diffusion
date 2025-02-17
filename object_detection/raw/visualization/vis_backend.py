from mmengine.visualization.vis_backend import AimVisBackend
from mmengine.config import Config
from mmengine.registry import VISBACKENDS

def list2dict(cfg):
    if isinstance(cfg, dict):
        cfg = {k: list2dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        cfg = {f"i{i}": list2dict(v) for i, v in enumerate(cfg)}
    
    return cfg

@VISBACKENDS.register_module()
class AimVisBackendv2(AimVisBackend):

    def add_config(self, config, **kwargs) -> None:
        """Record the config to Aim.
        Convert list to dictionaries.

        Args:
            config (Config): The Config object
        """
        if isinstance(config, Config):
            config = config.to_dict()
            # workaround because aim is not able to handle list in grouping
            config = list2dict(config)
        super().add_config(config, **kwargs)