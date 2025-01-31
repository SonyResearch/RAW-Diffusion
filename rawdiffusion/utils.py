import os
import torch
import numpy as np


def rggb_to_rgb(data):
    r, g1, g2, b = torch.chunk(data, 4, dim=1)
    g = (g1 + g2) / 2

    t = torch.cat(
        [r, g, b],
        dim=1,
    )
    return t


def rgb_to_rggb(
    data,
):
    r, g, b = torch.chunk(data, 3, dim=1)
    t = torch.cat(
        [r, g, g, b],
        dim=1,
    )
    return t


def rggb_to_rgb_np(data):
    r, g1, g2, b = np.split(data, 4, axis=-1)
    g = (g1 + g2) / 2
    return np.concatenate([r, g, b], axis=-1)


def gamma_correction(t, gamma=1.0 / 5):
    t = t.clip(0, 1)
    t = t**gamma
    return t


def create_folder_for_file(file_path):
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)


def parts_to_str(parts, delimiter="_"):
    return delimiter.join([str(p) for p in parts if p is not None])


def get_rgb_guidance_module_key(args):
    if args is None:
        return None

    model_name = args._target_.split(".")[-1]

    parts = [
        model_name,
    ]

    if model_name == "EDSR":
        parts += [args.n_resblocks, args.n_feats, args.bn]
    elif model_name == "RRDBNet":
        parts += [args.nf, args.nb]
    elif model_name == "NAFNet":
        parts += [
            args.width,
            "E" + parts_to_str(args.enc_blk_nums, "-"),
            args.middle_blk_num,
            "D" + parts_to_str(args.dec_blk_nums, "-"),
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return parts_to_str(parts)


def get_output_path(args):
    model_name = args.model._target_.split(".")[-1]
    model_params = args.model
    rgb_guidance_module_args = args.model.rgb_guidance_module
    rgb_guidance_module_key = get_rgb_guidance_module_key(rgb_guidance_module_args)
    train_split = os.path.splitext(os.path.basename(args.dataset.train.file_list))[
        0
    ].replace("_train", "")

    if args.dataset.train.max_items is not None:
        mi = args.dataset.train.max_items
        name, _ = os.path.splitext(args.dataset.train.file_list)
        train_split = f"{name}_{mi}_{args.general.seed}"

    parts = [
        os.path.basename(os.path.normpath(args.dataset.train.data_dir)),
        train_split,
        f"R{args.dataset.train.resample_dataset_size}"
        if args.dataset.train.resample_dataset_size is not None
        else None,
        args.general.image_size,
        args.dataset.train.batch_size,
        f"{args.general.max_steps // 1000}k",
        "rawdiffusion",
        args.diffusion.steps,
        args.diffusion.noise_schedule,
        "sigma" if args.diffusion.learn_sigma else None,
        "predict_noise" if not args.diffusion.predict_xstart else None,
        "model",
        model_name,
        args.model.model_channels,
        args.model.num_head_channels,
        args.model.num_res_blocks,
        args.model.norm_num_groups,
        "A" + parts_to_str(args.model.attention_resolutions, "-")
        if args.model.attention_resolutions
        else "noatt",
        "d{:.1f}".format(args.general.drop_rate)
        if args.general.drop_rate > 0.0
        else None,
        "ld{:.1f}".format(args.model.latent_drop_rate)
        if args.model.latent_drop_rate > 0.0
        else None,
        rgb_guidance_module_key,
        model_params.conditional_block_name
        if model_params.conditional_block_name != "RGBGuidedResidualBlock"
        else None,
        "midatt" if args.model.mid_attention else None,
        f"l2{args.general.weight_l2}" if args.general.weight_l2 > 0.0 else None,
        f"l1{args.general.weight_l1}" if args.general.weight_l1 > 0.0 else None,
        f"logl1{args.general.weight_logl1}"
        if args.general.weight_logl1 > 0.0
        else None,
        f"wd{args.general.weight_decay}" if args.general.weight_decay > 0.0 else None,
        args.general.lr_scheduler,
        "bl" if args.general.min_mode == "black_level" else "mv",
        "tanh" if args.model.out_tanh else None,
        args.general.suffix,
        args.general.seed,
    ]

    experiment_name = parts_to_str(parts)
    return os.path.join("experiments", experiment_name)
