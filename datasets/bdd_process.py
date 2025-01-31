import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.utils.bbox import BBox
from datasets.utils.coco_dataset import CocoDataset, ImageItem

classes = [
    "person",
    "bicycle",
    "car",
]

cityscapes_id_map = {
    "person": ["pedestrian", "rider"],
    "bicycle": ["bicycle"],
    "car": ["car"],
}

cs_id_to_class = {}
for c, ids in cityscapes_id_map.items():
    for i in ids:
        cs_id_to_class[i] = c

base_folder = "data/BDD/bdd100k"
output_root_path = "data/BDD_processed"

base_folder = os.path.expanduser(base_folder)

split = "train"
target_height = 416
target_width = 736

label_file = os.path.join(base_folder, f"labels/det_20/det_{split}.json")
image_folder = os.path.join(base_folder, "images/100k/", split)

dataset_name = f"bdd100k_h{target_height}"

base_destination_path = os.path.join(output_root_path, dataset_name)

target_images_base_path = os.path.join(base_destination_path, "images/100k/", split)
annotations_base_path = os.path.join(base_destination_path, "annotations")

data = json.load(open(label_file))

debug = False

ds = CocoDataset(classes)

dry_run = False

if debug:
    data = data[:10]


def resize(data, anns, target_height, target_width):
    ori_height, ori_width = data.shape[:2]

    if target_width is None:
        target_width = int(ori_width / ori_height * target_height)

    data = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_AREA)

    anns_transformed = BBox.resize_bboxes(
        anns,
        target_height=target_height,
        target_width=target_width,
        origin_height=ori_height,
        origin_width=ori_width,
    )

    return data, anns_transformed


def process_item(item):
    file_name = item["name"]
    labels = item.get("labels", [])
    print(f"processing {file_name}")

    image_path = os.path.join(image_folder, file_name)
    rel_path = os.path.relpath(image_path, image_folder)
    rel_path = os.path.splitext(rel_path)[0] + ".png"

    image_name = os.path.splitext(file_name)[0]

    img = Image.open(image_path)
    img = np.array(img)

    bboxes = []
    for label in labels:
        category = label["category"]
        box2d = label["box2d"]

        x1 = box2d["x1"]
        y1 = box2d["y1"]
        x2 = box2d["x2"]
        y2 = box2d["y2"]

        if category not in cs_id_to_class:
            continue

        category_mapped = cs_id_to_class[category]
        category_id = classes.index(category_mapped)

        bbox = BBox(
            x_min=x1,
            y_min=y1,
            x_max=x2,
            y_max=y2,
            category_id=category_id,
            image_id=image_name,
        )

        bboxes.append(bbox)

    if len(bboxes) == 0:
        return None, None, None

    # resize
    img, bboxes = resize(img, bboxes, target_height, target_width)

    image_item = ImageItem(
        file_name=rel_path,
        height=img.shape[0],
        width=img.shape[1],
        id=image_name,
    )
    target_path = os.path.join(target_images_base_path, rel_path)

    if not dry_run:
        # save
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        Image.fromarray(img).save(target_path)

    return target_path, image_item, bboxes


parallel = True
if parallel:
    with Pool(16) as p:
        results = p.map(process_item, data)
else:
    results = [process_item(item) for item in tqdm(data)]
rgb_paths = []

for rgb_path, image_item, bboxes in results:
    if rgb_path is None:
        continue

    # add to dataset
    ds.add_image_item(image_item)
    ds.add_annotations(bboxes)
    rgb_paths.append(rgb_path)

if not dry_run:
    name = f"{split}_100k.json"
    annotation_path = os.path.join(annotations_base_path, name)
    ds.save(annotation_path)

    txt_name = f"{split}_100k.txt"
    csv_path = os.path.join(base_destination_path, txt_name)
    with open(csv_path, "w") as f:
        for rgb_path in rgb_paths:
            rgb_path = os.path.relpath(rgb_path, base_destination_path)
            f.write(f",{rgb_path}\n")

print(f"{dataset_name} with {len(rgb_paths)} images created")
