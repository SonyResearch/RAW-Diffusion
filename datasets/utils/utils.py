import numpy as np
import math


def bayer_to_rggb(data):
    red, green_red, green_blue, blue = (
        data[0::2, 0::2],
        data[0::2, 1::2],
        data[1::2, 0::2],
        data[1::2, 1::2],
    )
    raw_arr = np.stack((red, green_red, green_blue, blue), axis=-1)
    return raw_arr


def patch_coordinates(
    image_height, image_width, num_patches, dividable_factor=1, overlap_factor=1
):
    assert num_patches > 1, "must be implemented for num_patches == 1"
    patch_width = (
        math.ceil(image_width / num_patches / dividable_factor * overlap_factor)
        * dividable_factor
    )
    patch_height = (
        math.ceil(image_height / num_patches / dividable_factor * overlap_factor)
        * dividable_factor
    )
    x_start_all = (np.linspace(0, 1, num_patches + 1)[:-1] * image_width).astype(
        np.int32
    )
    y_start_all = (np.linspace(0, 1, num_patches + 1)[:-1] * image_height).astype(
        np.int32
    )

    result = []

    for index_x, x_start in enumerate(x_start_all):
        for index_y, y_start in enumerate(y_start_all):
            x_end = x_start + patch_width
            y_end = y_start + patch_height

            if x_end > image_width:
                x_start = image_width - patch_width
                x_end = image_width

            if y_end > image_height:
                y_start = image_height - patch_height
                y_end = image_height

            result.append((index_x, index_y, x_start, y_start, x_end, y_end))

    return result
