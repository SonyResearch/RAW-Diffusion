import os
from glob import glob
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
    "person": [24, 25],  # person, rider
    "bicycle": [33],
    "car": [
        26,
    ],
}

cs_id_to_class = {}
for c, ids in cityscapes_id_map.items():
    for i in ids:
        cs_id_to_class[i] = c


dataset_path = "data/Cityscapes"
output_root_path = "data/Cityscapes_processed"

images_base_path = os.path.join(dataset_path, "leftImg8bit")
label_base_path = os.path.join(dataset_path, "gtFine")

target_height = 416
split = "train"
dataset_name = f"Cityscapes_h{target_height}"
base_destination_path = os.path.join(output_root_path, dataset_name)

target_images_base_path = os.path.join(base_destination_path, "leftImg8bit")
target_annotations_path = os.path.join(base_destination_path, "annotations")

os.makedirs(target_images_base_path, exist_ok=True)


files = glob(os.path.join(images_base_path, split, "*", "*.png"))
print(f"found {len(files)} files in {images_base_path}")

debug = False


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


ds = CocoDataset(classes)

if debug:
    files = files[:2]


def process_file(file_path):
    rel_path = os.path.relpath(file_path, images_base_path)
    print(f"processing {rel_path}")

    rel_label_path = rel_path.replace("leftImg8bit", "gtFine_instanceIds")
    img = Image.open(file_path)
    img = np.array(img)

    labels = Image.open(os.path.join(label_base_path, rel_label_path))
    labels = np.array(labels)

    img_name = os.path.splitext(os.path.basename(file_path))[0]

    bboxes = []

    for label in np.unique(labels):
        if label < 1000:
            # ignore stuff
            continue
        category = label // 1000
        if category not in cs_id_to_class:
            continue

        mask = labels == label
        y1, x1 = np.min(np.where(mask), axis=1)
        y2, x2 = np.max(np.where(mask), axis=1) + 1

        category_id = classes.index(cs_id_to_class[category])

        bbox = BBox(x1, y1, x2, y2, category_id=category_id, image_id=img_name)
        bboxes.append(bbox)

    if len(bboxes) == 0:
        None, None, None

    # resize
    img, bboxes = resize(img, bboxes, target_height, None)

    # add to dataset
    image_item = ImageItem(
        file_name=rel_path, height=img.shape[0], width=img.shape[1], id=img_name
    )

    # save
    target_path = os.path.join(target_images_base_path, rel_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    Image.fromarray(img).save(target_path)

    return target_path, image_item, bboxes


parallel = True

if parallel:
    with Pool(16) as p:
        results = p.map(process_file, files)
else:
    results = [process_file(file_path) for file_path in tqdm(files)]

rgb_paths = []
for rgb_path, image_item, bboxes in results:
    if rgb_path is None:
        continue

    ds.add_image_item(image_item)
    ds.add_annotations(bboxes)
    rgb_paths.append(rgb_path)

annotation_path = os.path.join(target_annotations_path, f"{split}.json")
ds.save(annotation_path)

txt_name = f"{split}.txt"
csv_path = os.path.join(base_destination_path, txt_name)
with open(csv_path, "w") as f:
    for rgb_path in rgb_paths:
        rgb_path = os.path.relpath(rgb_path, base_destination_path)
        f.write(f",{rgb_path}\n")

print(f"{dataset_name} created with {len(rgb_paths)} images")
