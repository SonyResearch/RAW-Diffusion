import random

import numpy as np


class ImageTransforms:
    def __init__(self, patch_size, is_train=True):
        self.patch_size = patch_size
        self.is_train = is_train

    def random_flip(self, raw_data, rgb_data):
        idx = np.random.randint(2)
        raw_data = np.flip(raw_data, axis=idx).copy()
        rgb_data = np.flip(rgb_data, axis=idx).copy()

        return raw_data, rgb_data

    def random_rotate(self, raw_data, rgb_data):
        idx = np.random.randint(4)
        raw_data = np.rot90(raw_data, k=idx)
        rgb_data = np.rot90(rgb_data, k=idx)

        return raw_data, rgb_data

    def random_crop(
        self,
        raw_data,
        rgb_data,
    ):
        image_height, image_width, _ = raw_data.shape
        rnd_h = random.randint(0, max(0, image_height - self.patch_size))
        rnd_w = random.randint(0, max(0, image_width - self.patch_size))

        patch_input_raw = raw_data[
            rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
        ]
        patch_rgb_data = rgb_data[
            rnd_h : rnd_h + self.patch_size, rnd_w : rnd_w + self.patch_size, :
        ]

        return patch_input_raw, patch_rgb_data

    def center_crop(self, raw_data, rgb_data):
        image_height, image_width, _ = raw_data.shape
        height_new = self.patch_size
        width_new = self.patch_size

        offset_y = (image_height - height_new) // 2
        offset_x = (image_width - width_new) // 2

        patch_input_raw = raw_data[
            offset_y : offset_y + height_new, offset_x : offset_x + width_new, :
        ]
        patch_rgb_data = rgb_data[
            offset_y : offset_y + height_new, offset_x : offset_x + width_new, :
        ]

        return patch_input_raw, patch_rgb_data

    def __call__(self, raw_data, rgb_data):
        assert raw_data.shape[:2] == rgb_data.shape[:2]

        if self.is_train:
            raw_data, rgb_data = self.random_crop(
                raw_data,
                rgb_data,
            )
            raw_data, rgb_data = self.random_rotate(raw_data, rgb_data)
            raw_data, rgb_data = self.random_flip(raw_data, rgb_data)
        else:
            raw_data, rgb_data = self.center_crop(raw_data, rgb_data)

        return raw_data, rgb_data
