import os

from torch.utils.data import DataLoader
from rawdiffusion.datasets.transforms import ImageTransforms
from .raw_image_dataset import RAWImageDataset
from rawdiffusion.datasets.camera import get_camera


def create_dataset(
    *,
    camera_name,
    data_dir,
    file_list,
    batch_size,
    seed,
    is_train=True,
    transform=True,
    permutate_once=False,
    resample_dataset_size=None,
    min_mode="black_level",
    patch_size=256,
    max_items=None,
    num_workers=8,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset_folder_name = os.path.basename(os.path.normpath(data_dir)).lower()
    dataset_name = dataset_folder_name.split("_")[0]
    if transform:
        transforms = ImageTransforms(
            patch_size=patch_size,
            is_train=is_train,
        )
    else:
        transforms = None

    assert min_mode in ["min_value", "black_level"]

    if dataset_name in ["fivek", "nod"]:
        camera = get_camera(dataset_name, camera_name)
        if camera is None:
            raise ValueError(f"unknown camera_name: {camera_name}")
        if min_mode == "min_value":
            raw_min_value = camera.min_value
        elif min_mode == "black_level":
            raw_min_value = camera.black_level
        else:
            raise ValueError(f"unknown min_mode: {min_mode}")

        raw_max_value = camera.white_level
    elif dataset_name in ["cityscapes", "bdd100k", "pascalraw"]:
        raw_min_value = None
        raw_max_value = None
    else:
        raise ValueError(f"unknown dataset_name: {dataset_name}")

    if max_items is not None:
        name, ext = os.path.splitext(file_list)
        file_list = f"{name}_{max_items}_{seed}{ext}"

    if "fivek" in dataset_folder_name:
        dataset = RAWImageDataset(
            file_list=file_list,
            dataset_path=data_dir,
            raw_min_value=raw_min_value,
            raw_max_value=raw_max_value,
            transforms=transforms,
        )
    elif "nod" in dataset_folder_name:
        dataset = RAWImageDataset(
            file_list=file_list,
            dataset_path=data_dir,
            raw_min_value=raw_min_value,
            raw_max_value=raw_max_value,
            transforms=transforms,
        )
    elif "cityscapes" in dataset_folder_name:
        dataset = RAWImageDataset(
            file_list=file_list,
            dataset_path=data_dir,
            raw_min_value=None,
            raw_max_value=None,
            transforms=transforms,
            rgb_only=True,
        )

    elif "bdd" in dataset_folder_name:
        dataset = RAWImageDataset(
            file_list=file_list,
            dataset_path=data_dir,
            raw_min_value=None,
            raw_max_value=None,
            transforms=transforms,
            rgb_only=True,
        )
    else:
        raise ValueError(f"unknown dataset_folder_name: {dataset_folder_name}")

    if permutate_once:
        from .dataset_wrapper import PermutedDataset

        dataset = PermutedDataset(dataset, seed=123)

    if resample_dataset_size is not None:
        from .dataset_wrapper import RandomSampleDataset

        dataset = RandomSampleDataset(dataset, n=resample_dataset_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )
    return loader
