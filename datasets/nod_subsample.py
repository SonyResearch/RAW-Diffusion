import argparse
import os
from pathlib import Path

import numpy as np

from datasets.utils.coco_dataset import CocoDataset

parser = argparse.ArgumentParser()
parser.add_argument("--max_images", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--dataset_path", type=str, default="data/NOD_processed/NOD_patches_3"
)

args = parser.parse_args()

max_images = args.max_images
seed = args.seed

dataset_path = args.dataset_path
annotation_path = os.path.join(dataset_path, "annotations")

original_annotation_path = Path("data/RAW-NOD/annotations")
subsampled_name = f"{max_images}_{seed}"

cameras = ["Nikon750", "Sony_RX100m7"]

if "patches" in dataset_path:
    num_patches = int(dataset_path.split("_")[-1])
    expected_images = max_images * num_patches**2
else:
    expected_images = max_images

has_annotation = "_patches" not in dataset_path

for camera in cameras:
    print(f"Processing {camera}")

    if "Nikon" in camera:
        json_path = original_annotation_path / "Nikon" / f"raw_new_{camera}_train.json"
    else:
        json_path = original_annotation_path / "Sony" / f"raw_new_{camera}_train.json"

    ds = CocoDataset.load(json_path)
    image_ids = [item.id for item in ds.images]
    image_ids.sort()

    print(f"Found {len(image_ids)} images in {json_path}")

    rng = np.random.default_rng(seed=seed)
    image_ids = rng.permutation(image_ids)
    image_ids = image_ids[:max_images]

    train_txt_file = os.path.join(dataset_path, f"{camera}_train.txt")
    with open(train_txt_file, "r") as f:
        txt_content = f.readlines()

    train_txt_subsampled_file = os.path.join(
        dataset_path, f"{camera}_train_{subsampled_name}.txt"
    )
    num_txt_images = 0
    with open(train_txt_subsampled_file, "w") as f:
        for line in txt_content:
            add_image = False
            for image_id in image_ids:
                if str(image_id) in line:
                    add_image = True
                    break
            if add_image:
                f.write(line)
                num_txt_images += 1

    if num_txt_images != expected_images:
        print(
            f"WARNING: Only {num_txt_images} images found in {train_txt_file} (expected {expected_images}))"
        )

    if has_annotation:
        annotation_prefixes = ["raw_new_", "rawpy_new_"]

        for annotation_prefix in annotation_prefixes:
            annotation_file = f"{annotation_prefix}{camera}_train.json"
            annotation_file_path = os.path.join(annotation_path, annotation_file)
            if not os.path.exists(annotation_file_path):
                print(f"Annotation file {annotation_file_path} does not exist")
                continue

            ann_ds = CocoDataset.load(annotation_file_path)
            ann_ds_subsampled = CocoDataset(ann_ds.categories)
            for image_id in image_ids:
                item = ann_ds.get_image_by_id(image_id)
                if item is None:
                    continue
                ann_ds_subsampled.images.append(item)
                ann_ds_subsampled.add_annotations(
                    ann_ds.get_image_annotations(image_id)
                )

            annotation_file_subsampled = (
                f"{annotation_prefix}{camera}_train_{subsampled_name}.json"
            )
            annotation_file_subsampled_path = os.path.join(
                annotation_path, annotation_file_subsampled
            )
            ann_ds_subsampled.save(annotation_file_subsampled_path)

            if len(ann_ds_subsampled.images) != expected_images:
                print(
                    f"WARNING: Only {len(ann_ds_subsampled.images)} images found in {annotation_file_subsampled_path} (expected {expected_images})"
                )

print(f"Dataset with {max_images} images created in {dataset_path}")
