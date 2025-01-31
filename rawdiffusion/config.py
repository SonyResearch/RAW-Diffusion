def mod_config(cfg):
    if cfg.model.channel_mult == "":
        if cfg.general.image_size == 512:
            cfg.model.channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif cfg.general.image_size == 256:
            cfg.model.channel_mult = (1, 1, 2, 2, 4, 4)
        elif cfg.general.image_size == 128:
            cfg.model.channel_mult = (1, 1, 2, 3, 4)
        elif cfg.general.image_size == 64:
            cfg.model.channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {cfg.general.image_size}")
    else:
        cfg.model.channel_mult = tuple(
            int(ch_mult) for ch_mult in cfg.model.channel_mult.split(",")
        )

    if cfg.model.attention_resolutions:
        cfg.model.attention_resolutions = cfg.model.attention_resolutions.split(",")

    cfg.model.out_channels = (
        cfg.model.out_channels
        if not cfg.diffusion.learn_sigma
        else 2 * cfg.model.out_channels
    )

    assert cfg.dataset.train.min_mode == cfg.dataset.val.min_mode, (
        "min_mode must be the same for train and val"
    )
