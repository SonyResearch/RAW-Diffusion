from typing import Optional
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from typing import List, Union
import os
import numpy as np

@DATASETS.register_module()
class NODDataset(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', ),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(0, 0, 142), (220, 20, 60),(106, 0, 228),]
    }

    def __init__(self,
                data_root: str,
                ann_file: str,
                ann_root: Optional[str] = None,
                dataset_name: Optional[str] = None,
                normalization=None,
                replace_filename=None,
                **kwargs) -> None:

        if dataset_name is not None:
            data_root = os.path.join(data_root, dataset_name)

        if ann_root is not None:
            ann_root = os.path.abspath(ann_root)
            ann_file = os.path.join(ann_root, ann_file)

        self.replace_filename = replace_filename

        self.normalization_data = None

        if normalization is not None:
            (min_source, max_source), (min_target, max_target) = normalization
                    
            scale = (max_target - min_target) / (max_source - min_source)
            offset = min_target - min_source * scale
            self.normalization_data = (scale, offset) # (scale, offset)

        print(f"normalization_data: {self.normalization_data}")

        super().__init__(data_root=data_root, ann_file=ann_file, **kwargs)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:

        raw_data_info = super().parse_data_info(raw_data_info)
        
        img_path = raw_data_info['img_path']

        if self.replace_filename:
            for k, v in self.replace_filename.items():
                img_path = img_path.replace(k, v)
        raw_data_info['img_path'] = img_path

        if self.normalization_data is not None:
            raw_data_info["normalize_reverse"] = self.normalization_data

        return raw_data_info
