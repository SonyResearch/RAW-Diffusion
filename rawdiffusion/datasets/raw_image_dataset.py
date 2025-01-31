from torch.utils.data import Dataset
import os
import imageio
import numpy as np
import torch


class RAWImageDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        file_list,
        raw_min_value=255,
        raw_max_value=16383,
        transforms=None,
        rgb_only=False,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.file_list = os.path.join(dataset_path, file_list)
        self.rgb_only = rgb_only

        self.data = self.load()
        self.transforms = transforms

        self.raw_min_value = raw_min_value
        self.raw_max_value = raw_max_value

    def load(self):
        data = []

        with open(self.file_list, "r") as f_read:
            item_list = [line.strip() for line in f_read.readlines()]

        for item in item_list:
            parts = item.split(",")
            assert len(parts) == 2, f"invalid item: {item}"
            raw_rel_path, rgb_rel_path = parts

            raw_path = os.path.join(self.dataset_path, raw_rel_path)
            rgb_path = os.path.join(self.dataset_path, rgb_rel_path)

            if (self.rgb_only or os.path.exists(raw_path)) and os.path.exists(rgb_path):
                data.append((raw_path, rgb_path))
            else:
                print(f"Warning: {raw_path} or {rgb_path} does not exist")

        return data

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2, 0, 1)

    def __getitem__(self, idx: int):
        raw_path, rgb_path = self.data[idx]

        if os.path.splitext(rgb_path)[1] == ".npy":
            rgb_data = np.load(rgb_path)
        else:
            rgb_data = imageio.imread(rgb_path)

        rgb_data = rgb_data.astype(np.float32) / 255

        if not self.rgb_only:
            raw_data_np = np.load(raw_path)
            raw_data = raw_data_np["raw"]
            raw_data = (raw_data.astype(np.float32) - self.raw_min_value) / (
                self.raw_max_value - self.raw_min_value
            )
        else:
            raw_data = np.zeros(
                (rgb_data.shape[0], rgb_data.shape[1], 4), dtype=np.float32
            )

        if rgb_data.shape[:2] != raw_data.shape[:2]:
            raise ValueError(
                f"target_rgb_img.shape: {rgb_data.shape}, input_raw_img.shape: {raw_data.shape}, file_name: {raw_path}, {rgb_path}"
            )

        if self.transforms is not None:
            raw_data, rgb_data = self.transforms(raw_data, rgb_data)

        raw_data = np.clip(raw_data, 0, 1)

        raw_data = self.np2tensor(raw_data).float()
        rgb_data = self.np2tensor(rgb_data).float()

        raw_data = raw_data * 2 - 1
        rgb_data = rgb_data * 2 - 1

        out_dict = {
            "raw_data": raw_data,
            "guidance_data": rgb_data,
            "path": os.path.relpath(rgb_path, self.dataset_path),
        }

        return out_dict

    def __len__(self):
        return len(self.data)
