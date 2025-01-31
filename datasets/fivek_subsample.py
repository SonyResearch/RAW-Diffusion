import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--max_images", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data/fivek_dataset_processed/fivek_patches_3",
)

args = parser.parse_args()

max_images = args.max_images
seed = args.seed

dataset_path = args.dataset_path
subsampled_name = f"{max_images}_{seed}"

cameras = ["Canon_EOS_5D", "NIKON_D700"]

if "patches" in dataset_path:
    num_patches = int(dataset_path.split("_")[-1])
    expected_images = max_images * num_patches**2
else:
    expected_images = max_images


for camera in cameras:
    print(f"Processing {camera}")

    file_list_path = os.path.join(dataset_path, f"{camera}_train.txt")
    image_ids = set()
    with open(file_list_path, "r") as f:
        file_list = f.readlines()
        for line in file_list:
            raw_file = line.split(",")[0]
            image_name = os.path.basename(raw_file)
            image_id = image_name.split("_")[0]
            image_ids.add(image_id)

    image_ids = list(image_ids)
    image_ids.sort()

    print(f"Found {len(image_ids)} images in {dataset_path}")

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

print(f"Dataset with {max_images} images created in {dataset_path}")
