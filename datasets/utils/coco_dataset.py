from pathlib import Path
import os
import json
from .bbox import BBox
from dataclasses import dataclass


@dataclass
class ImageItem:
    file_name: str
    height: int
    width: int
    id: int


class CocoDataset:
    def __init__(
        self,
        categories,
        images: list[ImageItem] | None = None,
        annotations: list[BBox] = None,
    ) -> None:
        self.images: list[ImageItem] = images or []
        self.annotations: list[BBox] = annotations or []
        self.next_bbox_id = (
            max([ann.bbox_id + 1 for ann in annotations]) if annotations else 1
        )

        assert len(categories) > 0
        if isinstance(categories[0], str):
            categories = [
                {"supercategory": "none", "id": obj_id, "name": obj_name}
                for obj_id, obj_name in enumerate(categories)
            ]

        self.categories = categories
        self.categoryIdToName = {c["id"]: c["name"] for c in categories}
        self.categoryNameToId = {c["name"]: c["id"] for c in categories}

    def __len__(self):
        return len(self.images)

    def add_image(self, filename, height, width, image_id):
        item = ImageItem(file_name=filename, height=height, width=width, id=image_id)
        self.images.append(item)

    def add_image_item(self, item: ImageItem):
        self.images.append(item)

    def add_annotation(self, ann: BBox):
        assert ann.image_id is not None

        if ann.bbox_id is None:
            ann.bbox_id = self.next_bbox_id
            self.next_bbox_id += 1

        self.annotations.append(ann)

    def add_annotations(self, anns: list[BBox]):
        for ann in anns:
            self.add_annotation(ann)

    def get_image_by_id(self, image_id):
        for item in self.images:
            if item.id == image_id:
                return item
        return None

    def get_image_annotations(self, image_id):
        return [ann for ann in self.annotations if ann.image_id == image_id]

    def save(self, output_json_path):
        output_json_path = Path(output_json_path)

        images_json = []

        for image in self.images:
            item = {
                "file_name": image.file_name,
                "height": image.height,
                "width": image.width,
                "id": image.id,
            }
            images_json.append(item)

        annotations_json = []
        for annotation in self.annotations:
            x_min = int(annotation.x_min)
            y_min = int(annotation.y_min)
            x_max = int(annotation.x_max)
            y_max = int(annotation.y_max)
            o_width = x_max - x_min + 1
            o_height = y_max - y_min + 1
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": annotation.image_id,
                "bbox": [x_min, y_min, o_width, o_height],
                "category_id": annotation.category_id,
                "id": annotation.bbox_id,
                "ignore": 0,
                "segmentation": [],
            }
            annotations_json.append(ann)

        json_data = {
            "images": images_json,
            "annotations": annotations_json,
            "categories": self.categories,
        }

        os.makedirs(output_json_path.parent, exist_ok=True)

        with open(output_json_path, "w") as outfile:
            outfile.write(json.dumps(json_data))

    @staticmethod
    def annotations_from_json(data):
        annotations = []
        for ann_json in data:
            x_min, y_min, o_width, o_height = ann_json["bbox"]
            x_max = x_min + o_width - 1
            y_max = y_min + o_height - 1
            category_id = ann_json["category_id"]
            bbox_id = ann_json["id"]
            image_id = ann_json["image_id"]
            ann = BBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                category_id=category_id,
                bbox_id=bbox_id,
                image_id=image_id,
            )
            annotations.append(ann)
        return annotations

    @staticmethod
    def images_from_json(data):
        images = []
        for img_json in data:
            file_name = img_json["file_name"]
            height = img_json["height"]
            width = img_json["width"]
            image_id = img_json["id"]
            item = ImageItem(
                file_name=file_name, height=height, width=width, id=image_id
            )
            images.append(item)
        return images

    @staticmethod
    def load(json_path):
        with open(json_path) as user_file:
            data = json.load(user_file)

        categories = data["categories"]
        images = data["images"]
        annotations = data["annotations"]

        images = CocoDataset.images_from_json(images)
        annotations = CocoDataset.annotations_from_json(annotations)

        # categoryIdToName = {c["id"]: c["name"] for c in categories}

        ds = CocoDataset(categories, images=images, annotations=annotations)

        return ds
