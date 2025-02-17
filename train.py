import os
from typing import Any

import hydra
import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import torch
import torch.optim.lr_scheduler as lr_scheduler
from aim.pytorch_lightning import AimLogger
from hydra.utils import instantiate
from lightning.pytorch.core import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torchinfo import summary
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from rawdiffusion.datasets.dataset_factory import create_dataset
from rawdiffusion.evaluation.collection import CollectionMetric
from rawdiffusion.evaluation.metrics import (
    MSEMetric,
    PearsonMetric,
    PSNRMetric,
    SSIMMetric,
)
from rawdiffusion.resample import create_named_schedule_sampler
from rawdiffusion.gaussian_diffusion_factory import (
    create_gaussian_diffusion,
)
from rawdiffusion.utils import get_output_path
from rawdiffusion.config import mod_config
from rawdiffusion.utils import rggb_to_rgb


class RAWDiffusionModule(LightningModule):
    def __init__(self, experiment_folder, **hparams) -> None:
        super().__init__()

        self.params = DictConfig(hparams)
        self.log_folder = experiment_folder
        self.save_hyperparameters()

        in_channels = self.params.model.in_channels
        image_size = self.params.general.image_size

        self.model = instantiate(self.params.model, image_size=image_size)
        self.diffusion = create_gaussian_diffusion(**self.params.diffusion)
        self.diffusion_val = create_gaussian_diffusion(**self.params.diffusion_val)
        self.schedule_sampler = create_named_schedule_sampler(
            self.params.general.schedule_sampler, self.diffusion
        )

        summary(
            self.model,
            input_size=[
                (1, in_channels, image_size, image_size),
                (1,),
                (1, 3, image_size, image_size),
            ],
            depth=2,
        )

    def normalize_inv(self, x):
        return (x + 1) / 2.0

    def setup(self, stage: str) -> None:
        self.logger.experiment["hparams"] = self.params

    def forward_step(self, input_data, guidance_input, sampling_seed=None):
        t, weights = self.schedule_sampler.sample(
            input_data.shape[0], self.device, seed=sampling_seed
        )

        losses, extra = self.diffusion.training_losses(
            self.model,
            input_data,
            t,
            model_kwargs=guidance_input,
            weight_l2=self.params.general.weight_l2,
            weight_l1=self.params.general.weight_l1,
            weight_logl1=self.params.general.weight_logl1,
        )

        loss = (losses["loss"] * weights).mean()
        metrics = {k: v * weights for k, v in losses.items() if k != "loss"}

        return loss, extra, metrics

    def training_step(self, batch, batch_idx):
        input_data = batch["raw_data"]
        guidance_data = batch["guidance_data"]

        guidance_input = self.preprocess_guidance(guidance_data)

        loss, extra, metrics = self.forward_step(input_data, guidance_input)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        for k, v in metrics.items():
            self.log(
                f"train_{k}",
                v.mean(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        if (
            self.global_step % self.params.general.log_train_images_interval == 0
            and self.global_step >= 0
        ):
            self.log_batch_results(guidance_input, extra)
        if (
            self.global_step % self.params.general.log_train_images_interval == 0
            and self.global_step >= 0
        ):
            self.log_sampling_images(batch)

        return loss

    def on_validation_start(self) -> None:
        self.metrics_sampling = CollectionMetric(
            {
                "mse_rggb": MSEMetric(),
                "psnr_rggb": PSNRMetric(),
                "ssim_rggb": SSIMMetric(),
                "peason_rggb": PearsonMetric(),
                "mse_rgb": MSEMetric(rggb_to_rgb=True),
                "psnr_rgb": PSNRMetric(rggb_to_rgb=True),
                "ssim_rgb": SSIMMetric(rggb_to_rgb=True),
                "peason_rgb": PearsonMetric(rggb_to_rgb=True),
            }
        )

        self.eval_diffusion_process = (
            self.current_epoch + 1
        ) % self.params.general.eval_diffusion_process_interval == 0
        print("validation_start", self.current_epoch, self.eval_diffusion_process)

    def validation_step(self, batch, batch_idx):
        input_data = batch["raw_data"]
        guidance_data = batch["guidance_data"]

        guidance_input = self.preprocess_guidance(guidance_data)

        sampling_seed = 123 + batch_idx
        loss, extra, metrics = self.forward_step(
            input_data, guidance_input, sampling_seed=sampling_seed
        )

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        val_steps_per_epoch = self.trainer.num_val_batches[0]
        sampling_interval = (
            val_steps_per_epoch // self.params.general.val_sampling_frequency
        )

        sampling = batch_idx % sampling_interval == 0
        sampling_log_images = (
            batch_idx // sampling_interval
        ) % self.params.general.log_val_sampling_images_interval == 0

        if sampling and sampling_log_images:
            filename = f"{batch_idx:04d}_{(self.global_step):06d}.png"
            self.log_batch_results(
                guidance_input, extra, mode="validation", filename=filename
            )

        if self.eval_diffusion_process and sampling:
            filename = f"{batch_idx:04d}_{(self.global_step):06d}.png"
            save_results = sampling_log_images
            raw_generated = self.log_sampling_images(
                batch,
                mode_name="validation_sampling",
                filename=filename,
                sampling_seed=sampling_seed,
                save_results=save_results,
            )
            input_data_device = input_data.to(raw_generated.device)

            self.metrics_sampling.update(
                self.normalize_inv(input_data_device), self.normalize_inv(raw_generated)
            )

    def on_validation_epoch_end(self) -> None:
        if self.eval_diffusion_process:
            for k, v in self.metrics_sampling.compute().items():
                self.log(f"val_sampling_{k}", v)

    def preprocess_guidance(self, guidance_data):
        guidance_input = {}

        drop_rate = self.params.general.drop_rate
        bs = guidance_data.shape[0]
        if self.training and drop_rate > 0.0:
            mask = (
                (torch.rand([bs, 1, 1, 1]) > drop_rate).float().to(guidance_data.device)
            )
            guidance_data = guidance_data * mask

        guidance_input["guidance_data"] = guidance_data

        return guidance_input

    def log_batch_results(
        self, model_kwargs, return_dict, mode="training", filename=None
    ):
        x_start = return_dict["x_start"]
        x_t = return_dict["x_t"]
        model_output = return_dict["model_output"]
        target = return_dict["target"]
        guidance_data = model_kwargs["guidance_data"]

        if x_start.shape[1] == 4:
            x_start = self.rggb_to_rgb_and_gc(x_start)
            x_t = self.rggb_to_rgb_and_gc(x_t)
            model_output = self.rggb_to_rgb_and_gc(model_output)
            target = self.rggb_to_rgb_and_gc(target)

        vis = torch.concatenate(
            [
                x_start,
                guidance_data.to(x_start.device),
                x_t,
                model_output,
                target,
                model_output - target,
            ],
            dim=3,
        )
        vis = torch.clamp(vis, -1, 1)
        vis = make_grid(vis, nrow=1)
        vis = self.normalize_inv(vis)

        if filename is None:
            filename = f"{(self.global_step):06d}.png"

        vis_path = os.path.join(self.log_folder, mode, filename)
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        to_pil_image(vis).save(vis_path)

    def log_sampling_images(
        self,
        batch,
        mode_name="training_sampling",
        filename=None,
        sampling_seed=None,
        save_results=True,
    ):
        input_data = batch["raw_data"]
        guidance_data = batch["guidance_data"]

        guidance_input = self.preprocess_guidance(guidance_data)

        bs, _, h, w = input_data.shape

        use_ddim = True
        clip_denoised = True
        diffusion = self.diffusion_val

        g = torch.Generator(device=self.device)
        if sampling_seed is not None:
            g.manual_seed(sampling_seed)

        with torch.inference_mode():
            shape = (bs, self.params.model.in_channels, h, w)
            noise = torch.randn(*shape, device=self.device, generator=g)

            sample_fn = (
                diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                model_kwargs=guidance_input,
                progress=True,
            )

        d = sample.device

        result_out = sample

        if save_results:
            if input_data.shape[1] == 4:
                input_data = self.rggb_to_rgb_and_gc(input_data)
                sample = self.rggb_to_rgb_and_gc(sample)

            vis = torch.concat([input_data.to(d), guidance_data.to(d), sample], dim=3)
            vis = self.normalize_inv(vis)

            vis = make_grid(vis, nrow=1)

            if filename is None:
                filename = f"{(self.global_step):06d}.png"

            vis_path = os.path.join(self.log_folder, mode_name, filename)
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            to_pil_image(vis).save(vis_path)

        return result_out

    @staticmethod
    def rggb_to_rgb_and_gc(data, gamma=1.0 / 5):
        data = rggb_to_rgb(data)

        data = (data + 1) / 2.0
        data = data**gamma
        data = data * 2 - 1

        return data

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            self.parameters(),
            lr=self.params.general.lr,
            weight_decay=self.params.general.weight_decay,
        )

        if self.params.general.lr_scheduler == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.params.general.max_steps,
            )
        elif self.params.general.lr_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.params.general.max_steps, eta_min=0.0
            )
        else:
            raise ValueError(
                f"Unknown lr_scheduler: {self.params.general.lr_scheduler}"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


@hydra.main(version_base="1.3", config_path="configs", config_name="rawdiffusion")
def main(cfg: DictConfig) -> None:
    mod_config(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.general.seed)

    aim_logger = AimLogger(
        experiment="diffusion",
        train_metric_prefix=None,
        test_metric_prefix=None,
        val_metric_prefix=None,
    )

    experiment_folder = get_output_path(cfg)
    print(f"experiment_folder: {experiment_folder}")

    print("creating data loader...")
    data_train = create_dataset(
        **cfg.dataset.train, seed=cfg.general.seed, patch_size=cfg.general.image_size
    )
    data_val = create_dataset(
        **cfg.dataset.val,
        seed=cfg.general.seed,
        patch_size=cfg.general.image_size,
        permutate_once=True,
    )

    raw_module = RAWDiffusionModule(experiment_folder=experiment_folder, **cfg)

    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.general.checkpoint:
        checkpoint_path = os.path.join(
            experiment_folder,
            "checkpoints",
        )
        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            save_last=True,
            every_n_train_steps=cfg.general.save_interval,
            enable_version_counter=False,
        )
        trainer_callbacks.append(checkpoint_cb)
        print("checkpoint", experiment_folder)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=cfg.general.max_steps,
        logger=aim_logger,
        callbacks=trainer_callbacks,
        enable_checkpointing=cfg.general.checkpoint,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epoch,
        limit_train_batches=1000,
    )

    trainer.fit(raw_module, data_train, data_val)


if __name__ == "__main__":
    main()
