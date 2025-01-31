import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from types import SimpleNamespace

import cv2
import exiftool
import numpy as np
import rawpy
from sklearn.model_selection import train_test_split

from datasets.utils.utils import patch_coordinates

parser = argparse.ArgumentParser()
parser.add_argument("--num_patches", type=int, default=3)
parser.add_argument("--dividable_factor", type=int, default=32)
parser.add_argument("--overlap", default=False, action="store_true")
parser.add_argument("--skip_train", default=False, action="store_true")

args = parser.parse_args()

################################################################
raw_source_root = "data/fivek_dataset"
save_folder_root = "data/fivek_dataset_processed/"

model_names = ["NIKON D700", "Canon EOS 5D"]
raw_file_extensions = [".dng"]

test_ratio = 0.15
seed = 2817
num_processes = 16
num_patches = args.num_patches
dividable_factor = args.dividable_factor
overlap = args.overlap
dry_run = False
debug = False
################################################################

dataset_name = "fivek"

if num_patches > 1:
    dataset_name += f"_patches_{num_patches}"

if overlap:
    dataset_name += "_overlap"

raw_source_root = os.path.expanduser(raw_source_root)
save_folder_root = os.path.join(save_folder_root, dataset_name)

raw_files_per_camera = defaultdict(list)


def get_camera_model_for_file(raw_path):
    raw_obj = SimpleNamespace(raw_path=raw_path)

    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(raw_path)
        raw_camera_model = metadata[0]["EXIF:Model"]
        orientation = metadata[0]["XMP:Orientation"]
        raw_obj.camera_model = raw_camera_model
        raw_obj.orientation = orientation
        print(raw_camera_model)

    return raw_obj


print("load meta data from raw files")
all_raw_files = []
for directory_path, _, files in os.walk(raw_source_root):
    for raw_file in files:
        file_ext = os.path.splitext(raw_file)[1]
        if file_ext in raw_file_extensions:
            raw_path = os.path.join(directory_path, raw_file)
            all_raw_files.append(raw_path)

with Pool(processes=32) as pool:
    raw_objects = pool.map(get_camera_model_for_file, all_raw_files)

for raw_object in raw_objects:
    raw_files_per_camera[raw_object.camera_model].append(raw_object)


def process_raw_file(raw_object, raw_destination_path, rgb_destination_path):
    raw_path = raw_object.raw_path
    orientation = raw_object.orientation
    camera_name = raw_object.camera_model

    file_name = os.path.splitext(os.path.basename(raw_path))[0]

    print(file_name)

    raw_file = rawpy.imread(raw_path)

    rgb_img = raw_file.postprocess()
    h, w, c = rgb_img.shape
    bayer_image = raw_file.raw_image_visible

    rgb_img_shrink = cv2.resize(
        rgb_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA
    ).astype(np.uint8)

    r_raw = bayer_image[0::2, 0::2][:, :, np.newaxis]
    g1_raw = bayer_image[0::2, 1::2][:, :, np.newaxis]
    g2_raw = bayer_image[1::2, 0::2][:, :, np.newaxis]
    b_raw = bayer_image[1::2, 1::2][:, :, np.newaxis]
    rggb_img = np.concatenate([r_raw, g1_raw, g2_raw, b_raw], axis=-1)

    cwb = raw_file.camera_whitebalance

    if orientation == 8:
        rggb_img = np.rot90(rggb_img, k=1)
    elif orientation == 6:
        rggb_img = np.rot90(rggb_img, k=3)
    elif orientation == 3:
        rggb_img = np.rot90(rggb_img, k=2)

    if camera_name == "Canon EOS 5D":
        rggb_img = np.maximum(rggb_img - 127.0, 0)

    rggb_img = rggb_img.astype(np.float32)

    raw_destination_paths = []
    rgb_destination_paths = []

    if num_patches > 1:
        img_height, img_width = rggb_img.shape[:2]
        overlap_factor = 2 if overlap else 1
        patch_coordinates_list = patch_coordinates(
            img_height,
            img_width,
            num_patches,
            dividable_factor=dividable_factor,
            overlap_factor=overlap_factor,
        )
        for index, (index_x, index_y, x_start, y_start, x_end, y_end) in enumerate(
            patch_coordinates_list
        ):
            rggb_img_patch = rggb_img[y_start:y_end, x_start:x_end]
            rgb_img_patch = rgb_img_shrink[y_start:y_end, x_start:x_end]
            raw_file_name = f"{file_name}_{index}.npz"
            rgb_file_name = f"{file_name}_{index}.npy"

            raw_np_path = os.path.join(raw_destination_path, raw_file_name)
            rgb_np_path = os.path.join(rgb_destination_path, rgb_file_name)

            if not dry_run:
                np.savez(raw_np_path, raw=rggb_img_patch, cwb=cwb)
                np.save(rgb_np_path, rgb_img_patch)

            raw_destination_paths.append(raw_np_path)
            rgb_destination_paths.append(rgb_np_path)

            if True:
                coordinate_filename = f"{file_name}_{index}.txt"
                coordinate_output_path = os.path.join(
                    raw_destination_path, coordinate_filename
                )
                with open(coordinate_output_path, "w") as f:
                    f.write(f"{x_start},{y_start},{x_end},{y_end}")

    else:
        raw_file_name = file_name + ".npz"
        rgb_file_name = file_name + ".npy"

        raw_np_path = os.path.join(raw_destination_path, raw_file_name)
        rgb_np_path = os.path.join(rgb_destination_path, rgb_file_name)

        if not dry_run:
            np.save(rgb_np_path, rgb_img_shrink)
            np.savez(raw_np_path, raw=rggb_img, cwb=cwb)

        raw_destination_paths.append(raw_np_path)
        rgb_destination_paths.append(rgb_np_path)

    return raw_destination_paths, rgb_destination_paths


for camera_model in model_names:
    print(f"process {camera_model}")

    raw_files = raw_files_per_camera[camera_model]
    raw_files.sort(key=lambda x: x.raw_path)
    print(f"{len(raw_files)} images")

    train_raw_files, test_raw_files = train_test_split(
        raw_files, test_size=test_ratio, random_state=seed
    )

    camera_folder_name = camera_model.replace(" ", "_")

    files_per_split = {"train": train_raw_files, "test": test_raw_files}

    for split_name in ("train", "test"):
        if split_name == "train" and args.skip_train:
            continue

        raw_files_split = files_per_split[split_name]
        print(f"{split_name} ({len(raw_files_split)} images)")
        destination_folder_name_raw = f"{camera_folder_name}_{split_name}_raw"
        destination_folder_name_rgb = f"{camera_folder_name}_{split_name}_rgb"

        raw_destination_path = os.path.join(
            save_folder_root, destination_folder_name_raw
        )
        rgb_destination_path = os.path.join(
            save_folder_root, destination_folder_name_rgb
        )

        os.makedirs(raw_destination_path, exist_ok=True)
        os.makedirs(rgb_destination_path, exist_ok=True)

        raw_paths = []
        rgb_paths = []

        parallel = True
        if debug:
            raw_files_split = raw_files_split[:10]

        f = partial(
            process_raw_file,
            raw_destination_path=raw_destination_path,
            rgb_destination_path=rgb_destination_path,
        )

        if parallel:
            with Pool(processes=num_processes) as pool:
                results = pool.map(f, raw_files_split)
                raw_paths, rgb_paths = zip(*results)
        else:
            for raw_file in raw_files_split:
                raw_path, rgb_path = f(
                    raw_file,
                )
                raw_paths.append(raw_path)
                rgb_paths.append(rgb_path)

        # flatten array
        raw_paths = [item for sublist in raw_paths for item in sublist]
        rgb_paths = [item for sublist in rgb_paths for item in sublist]

        raw_paths_rel = [
            os.path.relpath(raw_path, save_folder_root) for raw_path in raw_paths
        ]
        rgb_paths_rel = [
            os.path.relpath(rgb_path, save_folder_root) for rgb_path in rgb_paths
        ]

        csv_path = os.path.join(
            save_folder_root, f"{camera_folder_name}_{split_name}.txt"
        )
        if not dry_run:
            with open(csv_path, "w") as f:
                for raw_path, rgb_path in zip(raw_paths_rel, rgb_paths_rel):
                    f.write(f"{raw_path},{rgb_path}\n")
