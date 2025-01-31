from .coco_dataset import CocoDataset
import numpy as np
from .bbox import BBox
import rawpy


def load_one_raw(src_path):
    raw_file = rawpy.imread(str(src_path))
    black_level = np.mean(raw_file.black_level_per_channel)
    white_level = float(raw_file.white_level)
    rgb_img = raw_file.postprocess()

    bayer_image = raw_file.raw_image_visible
    return bayer_image, rgb_img, black_level, white_level, raw_file


def load_nod(json_path, data_root_path):
    ds = CocoDataset.load(json_path)

    def loader(image_idx):
        image_dict = ds.images[image_idx]
        image_id = image_dict.id
        file_name = image_dict.file_name
        annotations = ds.get_image_annotations(image_id)

        raw_path = data_root_path / file_name

        if "JPG" in file_name:
            raw_path = str(raw_path).replace("RawPy", "RAW")
            if "Sony" in raw_path:
                raw_path = raw_path.replace("JPG", "ARW")
            else:
                raw_path = raw_path.replace("JPG", "NEF")

            raw_data, rgb_img, black_level, white_level, raw_file = load_one_raw(
                raw_path
            )
            data = rgb_img
            is_raw = False
            raw_file = None
        else:
            data, rgb_img, black_level, white_level, raw_file = load_one_raw(raw_path)
            is_raw = True

        ann_height = image_dict.height
        ann_width = image_dict.width

        data_height, data_width = data.shape[:2]

        annotations = BBox.shift_bboxes(
            annotations,
            x_offset=(data_width - ann_width) / 2,
            y_offset=(data_height - ann_height) / 2,
        )

        result = {
            "image_id": image_id,
            "file_name": file_name,
            "is_raw": is_raw,
            "data": data,
            "annotations": annotations,
            "raw_file": raw_file,
        }

        return result

    return ds, loader
