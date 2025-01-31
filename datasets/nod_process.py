import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from datasets.utils.nod_loader import load_nod
from datasets.utils.utils import bayer_to_rggb, patch_coordinates

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
        "raw_new_Nikon750_val.json",
        "raw_new_Nikon750_test.json",
        "rawpy_new_Nikon750_train.json",
        "rawpy_new_Nikon750_val.json",
        "rawpy_new_Nikon750_test.json",
    ],
    "Sony": [
        "raw_new_Sony_RX100m7_train.json",
        "raw_new_Sony_RX100m7_val.json",
        "raw_new_Sony_RX100m7_test.json",
        "rawpy_new_Sony_RX100m7_test.json",
        "rawpy_new_Sony_RX100m7_val.json",
        "rawpy_new_Sony_RX100m7_train.json",
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("--num_patches", type=int, default=3)
parser.add_argument("--dividable_factor", type=int, default=32)
parser.add_argument("--overlap", default=False, action="store_true")
parser.add_argument("--skip_train", default=False, action="store_true")

args = parser.parse_args()

dry_run = False
num_patches = args.num_patches
dividable_factor = args.dividable_factor
overlap = args.overlap
debug = False
num_processes = 16


def process_item(i):
    result = loader(i)

    file_name = result["file_name"]
    data = result["data"]
    is_raw = result["is_raw"]
    raw_file = result["raw_file"]

    print(file_name)

    if is_raw:
        data = bayer_to_rggb(data)
        cwb = raw_file.camera_whitebalance
    else:
        data = cv2.resize(
            data, (data.shape[1] // 2, data.shape[0] // 2), interpolation=cv2.INTER_AREA
        )

    data_height, data_width = data.shape[:2]

    overlap_factor = 2 if overlap else 1
    patch_coordinates_list = patch_coordinates(
        data_height,
        data_width,
        num_patches,
        dividable_factor=dividable_factor,
        overlap_factor=overlap_factor,
    )

    item_output_files = []

    for index_x, index_y, x_start, y_start, x_end, y_end in patch_coordinates_list:
        patch_data = data[y_start:y_end, x_start:x_end]

        ext = "npz" if is_raw else "npy"

        data_filename = os.path.splitext(file_name)[0] + "_{}_{}.{}".format(
            index_x, index_y, ext
        )
        data_output_path = base_destination_path / split_name / data_filename
        item_output_files.append(data_output_path)

        if not dry_run:
            os.makedirs(data_output_path.parent, exist_ok=True)

            if is_raw:
                np.savez(data_output_path, raw=patch_data, cwb=cwb)
            else:
                np.save(data_output_path, patch_data)

        if not dry_run:
            coordinate_filename = os.path.splitext(file_name)[0] + "_{}_{}.txt".format(
                index_x, index_y
            )
            coordinate_output_path = (
                base_destination_path / split_name / coordinate_filename
            )
            with open(coordinate_output_path, "w") as f:
                f.write(f"{x_start},{y_start},{x_end},{y_end}")

    return item_output_files


for model, model_annotation_files in annotation_files.items():
    for ann_file in model_annotation_files:
        if "train" in ann_file and args.skip_train:
            continue
        print(model, ann_file)

        is_raw = "rawpy" not in ann_file

        data_subdir = "RawPy" if not is_raw else "RAW"
        ds_data_root_path = data_root_path / data_subdir / model
        ann_path = root_annotation_path / model / ann_file

        output_name = "NOD_patches_{}".format(num_patches)
        if overlap:
            output_name += "_overlap"

        base_destination_path = output_root_path / output_name

        ds, loader = load_nod(ann_path, data_root_path=ds_data_root_path)
        split_name = os.path.splitext(os.path.split(ann_file)[-1])[0]

        image_ids = range(len(ds))
        if debug:
            # random ids with seed
            rdn = np.random.RandomState(0)
            image_ids = rdn.choice(image_ids, 10, replace=False)
            # image_ids = image_ids[:10]
        all_output_files = []

        parallel = True

        f = partial(
            process_item,
        )

        if parallel:
            with Pool(processes=num_processes) as pool:
                all_output_files = pool.map(f, image_ids)
                all_output_files = [
                    item for sublist in all_output_files for item in sublist
                ]
        else:
            for raw_file in image_ids:
                item_output_files = f(
                    raw_file,
                )
                all_output_files.extend(item_output_files)

        if is_raw and not dry_run:
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
            with open(csv_path, "w") as f:
                for raw_path, rgb_path in zip(raw_paths_rel, rgb_paths_rel):
                    f.write(f"{raw_path},{rgb_path}\n")
