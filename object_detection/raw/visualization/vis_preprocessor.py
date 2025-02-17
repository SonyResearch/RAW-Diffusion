from mmengine import Registry
import numpy as np
import torch

# `scope` represents the domain of the registry. If not set, the default value is the package name.
# e.g. in mmdetection, the scope is mmdet
# `locations` indicates the location where the modules in this registry are defined.
# The Registry will automatically import the modules when building them according to these predefined locations.
VIS_PREPROCESSOR = Registry('vis_preprocessor')


def gamma_correction(img, gamma=0.25):
    if img.min() < 0:
        pass
    return img.astype(np.float32) ** gamma

def rggb_to_rgb(data): # normalize=True, 
    red, green_red, green_blue, blue = np.split(data, 4, axis=-1)
    green = (green_red + green_blue) / 2
    rgb = np.concatenate((red, green, blue), axis=-1)

    return rgb

def vis_bayer_heatmap(bayer_img):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('jet')

    bayer_heatmap = cmap(bayer_img / np.max(bayer_img))
    bayer_heatmap = np.delete(bayer_heatmap, 3, 2)

    return bayer_heatmap

@VIS_PREPROCESSOR.register_module()
class DataToRGB:
    def __init__(self, value_min=None, value_max=None, raw_gamma=None, clip=True):
        super().__init__()
        self.raw_gamma = raw_gamma
        self.value_max = value_max
        self.value_min = value_min
        self.clip = clip

    def transform(self, item, batch_idx=None):
        if isinstance(item, dict):
            img = item['inputs']
        else:
            img = item
        
        
        if batch_idx is not None:
            img = img[batch_idx]

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().detach().numpy()

        assert len(img.shape) == 3

        data_channel_dim = img.shape[2]

        vmin = self.value_min
        if vmin is None:
            vmin = 0
        if self.value_max is not None:
            img = (img.astype(np.float32) - vmin) / (self.value_max - vmin) * 255

        if data_channel_dim == 1:
            img  = vis_bayer_heatmap(img.squeeze(2) / 255) * 255

        elif data_channel_dim == 3:
            # rgb image
            pass
        elif data_channel_dim == 4:
            # 4-channel raw
            img = rggb_to_rgb(img)

        if self.clip:
            img = np.clip(img, 0, 255)

        else:
            raise ValueError(f"unknown data format, got {img.shape}")

        if self.raw_gamma is not None:
            img = gamma_correction(img/255, gamma=self.raw_gamma)*255

        return img
