import math
import multiprocessing
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from datasets.utils.bbox import BBox
from datasets.utils.coco_dataset import CocoDataset, ImageItem
from datasets.utils.nod_loader import load_nod
from datasets.utils.utils import bayer_to_rggb

data_root_path = Path("data/NOD/")
root_annotation_path = Path("data/RAW-NOD/annotations")
output_root_path = Path("data/NOD_processed/")

if not os.path.exists(root_annotation_path):
    raise ValueError(
        "Please download the annotations from https://github.com/igor-morawski/RAW-NOD"
    )

annotation_files = {
    "Nikon": [
        "raw_new_Nikon750_train.json",
        # "raw_new_Nikon750_val.json",
        "raw_new_Nikon750_test.json",
        "rawpy_new_Nikon750_train.json",
        # "rawpy_new_Nikon750_val.json",
        "rawpy_new_Nikon750_test.json",
    ],
    "Sony": [
        "raw_new_Sony_RX100m7_train.json",
        # "raw_new_Sony_RX100m7_val.json",
        "raw_new_Sony_RX100m7_test.json",
        "rawpy_new_Sony_RX100m7_test.json",
        # "rawpy_new_Sony_RX100m7_val.json",
        "rawpy_new_Sony_RX100m7_train.json",
    ],
}


def resize(data, anns, target_height, target_width):
    ori_height, ori_width = data.shape[:2]

    if target_width is None:
        target_width = int(ori_width / ori_height * target_height)

    if dividable_factor is not None:
        target_width = math.ceil(target_width / dividable_factor) * dividable_factor

    data = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_AREA)

    anns_transformed = BBox.resize_bboxes(
        anns,
        target_height=target_height,
        target_width=target_width,
        origin_height=ori_height,
        origin_width=ori_width,
    )

    return data, anns_transformed


h_target = 400
w_target = None
dividable_factor = 32
dry_run = False
debug = False

if dividable_factor is not None:
    h_target = math.ceil(h_target / dividable_factor) * dividable_factor


def process_item(i):
    result = loader(i)

    image_id = result["image_id"]
    file_name = result["file_name"]
    data = result["data"]
    annotations = result["annotations"]
    is_raw = result["is_raw"]
    raw_file = result["raw_file"]
    print(file_name)

    if is_raw:
        data = bayer_to_rggb(data)
        annotations = BBox.scale_bboxes(annotations, factor=0.5)

        cwb = raw_file.camera_whitebalance

    data, annotations = resize(
        data, annotations, target_height=h_target, target_width=w_target
    )

    ext = "npz" if is_raw else "npy"
    data_filename = os.path.splitext(file_name)[0] + f".{ext}"
    data_height, data_width = data.shape[:2]

    image_item = ImageItem(
        file_name=data_filename,
        height=data_height,
        width=data_width,
        id=image_id,
    )

    data_output_path = base_destination_path / split_name / data_filename

    if not dry_run:
        os.makedirs(data_output_path.parent, exist_ok=True)
        if is_raw:
            np.savez(data_output_path, raw=data, cwb=cwb)
        else:
            np.save(data_output_path, data)

    return data_output_path, image_item, annotations


for model, model_annotation_files in annotation_files.items():
    for ann_file in model_annotation_files:
        print(model, ann_file)
        is_raw = "raw_new" in ann_file
        data_subdir = "RawPy" if not is_raw else "RAW"
        ds_data_root_path = data_root_path / data_subdir / model
        ann_path = root_annotation_path / model / ann_file

        if w_target is not None:
            output_name = "NOD_h{}_w{}".format(h_target, w_target)
        else:
            output_name = "NOD_h{}".format(h_target)

        if dividable_factor is not None:
            output_name += "_d{}".format(dividable_factor)

        base_destination_path = output_root_path / output_name

        ds, loader = load_nod(ann_path, data_root_path=ds_data_root_path)
        split_name = os.path.splitext(os.path.split(ann_file)[-1])[0]
        ds_processed = CocoDataset(ds.categories)

        image_ids = range(len(ds))

        if debug:
            image_ids = image_ids[:10]

        all_output_files = []

        parallel = True
        if parallel:
            with multiprocessing.Pool(16) as p:
                results = p.map(process_item, image_ids)

        else:
            results = [process_item(i) for i in tqdm(image_ids)]

        for data_output_path, image_item, annotations in results:
            ds_processed.add_image_item(image_item)
            ds_processed.add_annotations(annotations)
            all_output_files.append(data_output_path)

        json_name = f"{split_name}.json"
        output_json_path = base_destination_path / "annotations" / json_name

        if not dry_run:
            ds_processed.save(output_json_path)

        if is_raw:
            # because RAW and RGB are processed individually, we write the joint raw-rgb pairs file once when processing the raw images

            rgb_split_name = split_name.replace("raw", "rawpy")
            rgb_paths = []
            for data_output_path in all_output_files:
                data_output_path_rgb = (
                    data_output_path.parent.parent
                    / rgb_split_name
                    / data_output_path.name.replace(".npz", ".npy")
                )
                rgb_paths.append(data_output_path_rgb)

            raw_paths_rel = [
                raw_path.relative_to(base_destination_path)
                for raw_path in all_output_files
            ]
            rgb_paths_rel = [
                rgb_path.relative_to(base_destination_path) for rgb_path in rgb_paths
            ]

            txt_name = split_name.replace("raw_new_", "") + ".txt"
            csv_path = os.path.join(base_destination_path, txt_name)
            if not dry_run:
                with open(csv_path, "w") as f:
                    for raw_path, rgb_path in zip(raw_paths_rel, rgb_paths_rel):
                        f.write(f"{raw_path},{rgb_path}\n")
