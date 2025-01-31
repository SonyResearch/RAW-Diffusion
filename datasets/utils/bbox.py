import copy


class BBox:
    def __init__(
        self, x_min, y_min, x_max, y_max, category_id=None, image_id=None, bbox_id=None
    ) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.category_id = category_id
        self.image_id = image_id
        self.bbox_id = bbox_id

    @property
    def width(self):
        return self.x_max - self.x_min + 1

    @property
    def height(self):
        return self.y_max - self.y_min + 1

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return f"BBox(cat: {self.category_id}, bbox: ({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}))"

    @staticmethod
    def resize_bboxes(
        bbox_list: list["BBox"],
        target_height,
        target_width,
        origin_height,
        origin_width,
    ):
        def x_trans(x):
            return x / (origin_width - 1) * (target_width - 1)

        def y_trans(y):
            return y / (origin_height - 1) * (target_height - 1)

        return BBox.transform_bboxes(bbox_list, x_trans=x_trans, y_trans=y_trans)

    @staticmethod
    def shift_bboxes(bbox_list: list["BBox"], y_offset, x_offset):
        def x_trans(x):
            return x + x_offset

        def y_trans(y):
            return y + y_offset

        return BBox.transform_bboxes(bbox_list, x_trans=x_trans, y_trans=y_trans)

    @staticmethod
    def scale_bboxes(bbox_list: list["BBox"], factor):
        def x_trans(x):
            return x * factor

        def y_trans(y):
            return y * factor

        return BBox.transform_bboxes(bbox_list, x_trans=x_trans, y_trans=y_trans)

    @staticmethod
    def transform_bboxes(bbox_list: list["BBox"], x_trans, y_trans):
        bbox_list: list[BBox] = copy.deepcopy(bbox_list)

        for bbox in bbox_list:
            bbox.x_min = x_trans(bbox.x_min)
            bbox.x_max = x_trans(bbox.x_max)
            bbox.y_min = y_trans(bbox.y_min)
            bbox.y_max = y_trans(bbox.y_max)

        return bbox_list
