"""
train_base.py

MimicMotion(UNet + PoseNet) 자체를 학습하는 베이스라인 학습 스크립트.
VAE / CLIP image encoder는 고정하고, diffusion UNet과 PoseNet을 EDM(SVD)
스타일의 denoising loss로 직접 학습한다.

실행:
    python train_base.py --config configs/train_config.yaml
"""

from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import itertools
import logging
import math

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MimicMotionFramesDataset, collate_fn
from utils import (
    ensure_dir,
    encode_video_latents,
    encode_reference_image_embeds,
    encode_reference_image_latents,
    encode_pose_latents,
    get_added_time_ids,
    load_components,
    set_seed,
)

LOGGER = logging.getLogger("train_base")


def train(cfg: OmegaConf) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir  = ensure_dir(cfg.output_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    writer   = SummaryWriter(str(ensure_dir(out_dir / "tensorboard")))

    train_ds = MimicMotionFramesDataset(
        str(cfg.train_manifest), int(cfg.resolution), int(cfg.num_frames),
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.batch_size), shuffle=True,
        num_workers=int(cfg.num_workers), collate_fn=collate_fn, drop_last=True,
    )

    overfit_single = bool(cfg.get("overfit_single_sample", False))
    fixed_batch = None
    if overfit_single:
        fixed_batch = next(iter(train_loader))
        LOGGER.info("Overfit mode: ON")

    overfit_fixed_sigma = bool(cfg.get("overfit_fixed_sigma", False))
    if overfit_fixed_sigma:
        LOGGER.info("Fixed-sigma mode: ON")

    init_ckpt = str(cfg.init_checkpoint) if cfg.get("init_checkpoint") else None
    comps = load_components(str(cfg.base_model_path), device, init_checkpoint=init_ckpt)
    comps.vae.requires_grad_(False)
    comps.vae.eval()
    comps.image_encoder.requires_grad_(False)
    comps.image_encoder.eval()
    comps.unet.train()
    comps.pose_net.requires_grad_(False)
    comps.pose_net.eval()
    unet_dtype = next(comps.unet.parameters()).dtype

    trainable_params = list(itertools.chain(
        comps.unet.parameters(),
    ))
    total_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    LOGGER.info("Trainable params (unet): %d (%.2f M)",
                total_params, total_params / 1e6)

    optimizer = AdamW(trainable_params,
                      lr=float(cfg.get("learning_rate", 1e-5)),
                      betas=(0.9, 0.999), weight_decay=1e-2)
    warmup = int(cfg.get("lr_warmup_steps", 500))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / warmup))

    noise_aug_strength = float(cfg.noise_aug_strength)
    sigma_data = float(cfg.get("sigma_data", 1.0))
    p_mean = float(cfg.get("p_mean", 0.0))
    p_std  = float(cfg.get("p_std",  0.7))
    sigma_sample_mode = str(cfg.get("sigma_sample_mode", "lognormal"))
    sigma_min = float(cfg.get("sigma_min", 0.002))
    sigma_max = float(cfg.get("sigma_max", 700.0))
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    if sigma_sample_mode == "loguniform":
        LOGGER.info("sigma sampling: loguniform [%.4f, %.1f]", sigma_min, sigma_max)
    grad_clip = float(cfg.get("grad_clip", 1.0))

    global_step = 0
    fixed_noise_cache: dict = {}

    try:
        num_epochs = int(cfg.num_epochs)
        total_steps = num_epochs * len(train_loader)
        for _ in range(num_epochs):
            comps.unet.train()
            epoch_iter = ((fixed_batch for _ in range(len(train_loader)))
                          if overfit_single else train_loader)

            for batch in epoch_iter:
                pixel_values = batch["pixel_values"].to(device)
                pose_images  = batch["pose_images"].to(device)
                ref_image    = batch["ref_image"].to(device)
                b, f = pixel_values.shape[:2]

                with torch.no_grad():
                    video_latents = encode_video_latents(comps, pixel_values)
                    ref_embeds    = encode_reference_image_embeds(comps, ref_image)

                    def _sample_sigma():
                        if sigma_sample_mode == "loguniform":
                            u = torch.rand([b, 1, 1, 1, 1], device=device)
                            return (u * (log_sigma_max - log_sigma_min) + log_sigma_min).exp()
                        rnd = torch.randn([b, 1, 1, 1, 1], device=device)
                        return (rnd * p_std + p_mean).exp()

                    if overfit_fixed_sigma:
                        if not fixed_noise_cache:
                            fixed_noise_cache["sigma"] = _sample_sigma()
                            fixed_noise_cache["noise"] = torch.randn_like(video_latents)
                        sigma = fixed_noise_cache["sigma"]
                        noise = fixed_noise_cache["noise"]
                    else:
                        sigma = _sample_sigma()
                        noise = torch.randn_like(video_latents)

                    ref_cond_latents = encode_reference_image_latents(
                        comps, ref_image,
                        noise_aug_strength=noise_aug_strength, noise_for_aug=noise,
                    )
                    ref_image_latents = ref_cond_latents.unsqueeze(1).repeat(1, f, 1, 1, 1)

                    sigma_sq = sigma ** 2
                    c_in    = 1.0 / (sigma_sq + sigma_data ** 2) ** 0.5
                    c_noise = sigma.log() / 4.0
                    noisy_latents = video_latents + noise * sigma
                    input_latents = c_in * noisy_latents
                    latent_input  = torch.cat([input_latents, ref_image_latents], dim=2).to(unet_dtype)
                    added_time_ids = get_added_time_ids(
                        int(cfg.fps) - 1, int(cfg.motion_bucket_id),
                        noise_aug_strength, b, device, unet_dtype,
                    )
                    c_noise_unet = c_noise.reshape(b).to(unet_dtype)

                # pose_net runs with grad since it is being trained
                pose_latents = encode_pose_latents(comps, pose_images)

                model_pred = comps.unet(
                    latent_input, c_noise_unet,
                    encoder_hidden_states=ref_embeds.to(unet_dtype),
                    added_time_ids=added_time_ids,
                    pose_latents=pose_latents.to(unet_dtype),
                    image_only_indicator=False,
                ).sample.float()

                # v-prediction preconditioning, matching this scheduler's actual
                # config (prediction_type: v_prediction) and EulerDiscreteScheduler
                # .step()'s own formula exactly:
                #   pred_original_sample = model_output * (-sigma / sqrt(sigma^2+1))
                #                          + sample / (sigma^2+1)
                # c_out is NEGATIVE here. Using the generic (positive) EDM F_theta
                # sign instead trains model_pred in the opposite direction from what
                # scheduler.step() actually needs, which silently destroys the
                # pretrained UNet within a few hundred steps (validated 2026-07-03:
                # loss looked fine but sampled output collapsed to pure noise).
                c_skip = sigma_data ** 2 / (sigma_sq + sigma_data ** 2)
                c_out  = -sigma * sigma_data / (sigma_sq + sigma_data ** 2) ** 0.5
                loss_weight = (sigma_sq + sigma_data ** 2) / (sigma * sigma_data) ** 2

                denoised = model_pred * c_out + c_skip * noisy_latents
                loss = (loss_weight * (denoised - video_latents) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
                lr_scheduler.step()

                if global_step % int(cfg.print_every_steps) == 0:
                    progress_pct = 100.0 * global_step / total_steps
                    LOGGER.info("Step %d/%d (%.2f%%) | loss=%.5f  lr=%.2e",
                                global_step, total_steps, progress_pct,
                                loss.item(), lr_scheduler.get_last_lr()[0])
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/progress_pct", progress_pct, global_step)

                if global_step > 0 and global_step % int(cfg.save_every_steps) == 0:
                    sp = ckpt_dir / f"step_{global_step:08d}.pth"
                    torch.save(_trainable_state_dict(comps), sp)
                    LOGGER.info("Saved: %s", sp)

                global_step += 1

    finally:
        fp = ckpt_dir / f"last_{global_step:08d}.pth"
        torch.save(_trainable_state_dict(comps), fp)
        LOGGER.info("Final: %s", fp)
        writer.close()


def _trainable_state_dict(comps) -> dict:
    """Flat, prefixed state dict matching the layout MimicMotionModel.load_state_dict
    (mimicmotion/utils/loader.py) expects, e.g. 'unet.xxx', 'pose_net.xxx'."""
    ckpt = {}
    ckpt.update({f"unet.{k}": v for k, v in comps.unet.state_dict().items()})
    ckpt.update({f"pose_net.{k}": v for k, v in comps.pose_net.state_dict().items()})
    return ckpt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/train_config.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
