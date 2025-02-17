from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import FileClient
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

from mmcv.transforms import LoadImageFromFile
from PIL import Image
import os

@TRANSFORMS.register_module()
class LoadNumpyFromFile(LoadImageFromFile):
    # inherit from LoadImageFromFile e.g. for get_loading_pipeline() detection


    """Load an image from file.
    Required Keys:
    - img_path
    Modified Keys:
    - img
    - img_shape
    - ori_shape
    """

    def __init__(self,
                 to_float32: bool = True,
                 ) -> None:
        super().__init__()
        self.to_float32 = to_float32


    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.
        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        ext = os.path.splitext(filename)[-1].lower()

        if ext == ".png":
            img = Image.open(filename)
            img = np.array(img).astype(np.float32) # / 255.0
        elif ext in (".h5", ".hdf5"):
            import h5py
            with h5py.File(filename, 'r') as f:
                img = f["raw"][:]
        else:
            img = np.load(filename)
            if filename.endswith(".npz"):
                img = img["raw"]

        if self.to_float32:
            img = img.astype(np.float32)

        if "normalize_reverse" in results:
            scale, offset = results["normalize_reverse"]
            img = img * scale + offset

        if "upsample_factor" in results:
            from mmcv.image import imrescale
            img = imrescale(img, results["upsample_factor"])

        if "rgb2rggb" in results:
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            img = np.stack([r, g, g, b], axis=-1)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, ')

        return repr_str