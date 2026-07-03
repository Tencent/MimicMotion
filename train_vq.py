"""
train_vq.py

train_base.py의 EDM(SVD) denoising 학습 baseline에 spatio-temporal continuous
tokenizer(patchify -> linear encode -> linear decode -> unpatchify) auxiliary
branch를 추가한 버전. pose encoder/FiLM/VQ/codebook은 아직 구현하지 않으며,
tokenizer의 reconstruction은 UNet에 주입되지 않는다 (`enable_continuous_tokenizer`
가 False면 baseline과 100% 동일하게 동작).

실행:
    python train_vq.py --config configs/train_config.yaml
"""

from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import logging
import math

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MimicMotionFramesDataset, collate_fn
from feature_inspect import FeatureInspector
from visual_tokenizer import (
    SpatioTemporalContinuousTokenizer,
    UNetFeatureCapture,
    is_rank0,
    restore_video_dimension,
)
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

    debug_overfit_single_batch = bool(cfg.get("debug_overfit_single_batch", False))
    debug_overfit_steps = int(cfg.get("debug_overfit_steps", 500))

    overfit_single = bool(cfg.get("overfit_single_sample", False)) or debug_overfit_single_batch
    fixed_batch = None
    if overfit_single:
        fixed_batch = next(iter(train_loader))
        LOGGER.info("Overfit mode: ON")
    if debug_overfit_single_batch:
        LOGGER.info("Debug single-batch overfit mode: ON (%d steps, then stop and report)",
                    debug_overfit_steps)

    overfit_fixed_sigma = bool(cfg.get("overfit_fixed_sigma", False))
    if overfit_fixed_sigma:
        LOGGER.info("Fixed-sigma mode: ON")
    if debug_overfit_single_batch and not overfit_fixed_sigma:
        LOGGER.warning(
            "debug_overfit_single_batch=true but overfit_fixed_sigma=false: sigma/noise "
            "are re-sampled every step, so the down_blocks[1] feature the tokenizer sees "
            "is NOT actually fixed across steps (validated 2026-07-03: reconstruction_loss "
            "barely moved / got worse over 60 steps under this combination). The single-batch "
            "overfit report below will likely be misleading -- set overfit_fixed_sigma: true."
        )

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

    # Read-only, one-shot feature shape logging for deciding where a future
    # VQ/FiLM/pose-conditioned module would attach. Hooks remove themselves
    # after the first forward pass; no effect on training when disabled or
    # once inspection has fired. See feature_inspect.py.
    FeatureInspector(
        comps.unet, comps.pose_net,
        enabled=bool(cfg.get("enable_feature_inspection", False)),
    )

    # Spatio-temporal continuous tokenizer (auxiliary branch, no pose
    # encoder/FiLM/VQ yet). Selected feature = down_blocks[1] output, per the
    # feature inspection done on 2026-07-03 (see visual_tokenizer.py
    # docstring). reconstructed feature is NOT injected back into the UNet in
    # this stage.
    enable_continuous_tokenizer = bool(cfg.get("enable_continuous_tokenizer", False))
    visual_tokenizer = None
    feature_capture = None

    if enable_continuous_tokenizer:
        visual_tokenizer = SpatioTemporalContinuousTokenizer(
            in_channels=comps.unet.config.block_out_channels[1],
            embed_dim=int(cfg.get("visual_embed_dim", 64)),
            temporal_patch_size=int(cfg.get("temporal_patch_size", 4)),
            spatial_patch_height=int(cfg.get("spatial_patch_height", 4)),
            spatial_patch_width=int(cfg.get("spatial_patch_width", 3)),
        ).to(device)
        feature_capture = UNetFeatureCapture(comps.unet.down_blocks[1])
        feature_name = str(cfg.get("visual_feature_name", "down_blocks[1].output"))
        LOGGER.info("Continuous tokenizer enabled on feature=%s", feature_name)

    # baseline's own trainable-parameter setup (unet) is left exactly as-is;
    # the tokenizer is only ever ADDED as its own optimizer param group.
    param_groups = [{"params": list(comps.unet.parameters()), "lr": float(cfg.get("learning_rate", 1e-5))}]
    if enable_continuous_tokenizer:
        tokenizer_lr = float(cfg.tokenizer_learning_rate)
        param_groups.append({"params": list(visual_tokenizer.parameters()), "lr": tokenizer_lr})

    trainable_params = [p for group in param_groups for p in group["params"]]
    total_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    tokenizer_param_count = sum(p.numel() for p in visual_tokenizer.parameters()) if enable_continuous_tokenizer else 0
    LOGGER.info("Trainable params: %d (%.2f M)  [unet=%d, tokenizer=%d]",
                total_params, total_params / 1e6,
                sum(p.numel() for p in comps.unet.parameters()), tokenizer_param_count)

    optimizer = AdamW(param_groups,
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
    debug_metrics = {"initial": None, "final": None}

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

                # Continuous tokenizer auxiliary branch. reconstructed feature is
                # NOT fed back into the UNet (feature_for_unet stays the
                # original tensor the UNet already produced); this is purely
                # an additional loss term. Disabled reproduces the baseline
                # loss exactly (total_loss is loss).
                total_loss = loss
                tokenizer_output = None
                if enable_continuous_tokenizer:
                    raw_feature = feature_capture.output
                    if raw_feature is None:
                        raise RuntimeError(
                            "continuous_tokenizer: feature capture hook on "
                            "comps.unet.down_blocks[1] produced no output; "
                            "the UNet forward call above did not run as expected."
                        )
                    feature_5d = restore_video_dimension(raw_feature, batch_size=b, num_frames=f)
                    if bool(cfg.get("detach_tokenizer_input", True)):
                        feature_5d = feature_5d.detach()
                    # model_pred above already ran on the original feature;
                    # tokenizer_output is never fed back into the UNet here.
                    tokenizer_output = visual_tokenizer(feature_5d)
                    lambda_recon = float(cfg.get("lambda_tokenizer_reconstruction", 1.0))
                    total_loss = loss + lambda_recon * tokenizer_output["reconstruction_loss"]

                    if global_step == 0 and is_rank0():
                        LOGGER.info(
                            "[ContinuousTokenizer shapes] selected_feature=%s restored_5d=%s "
                            "patch=%s token=%s reconstructed_feature=%s",
                            tuple(raw_feature.shape), tuple(feature_5d.shape),
                            tuple(tokenizer_output["patches"].shape),
                            tuple(tokenizer_output["tokens"].shape),
                            tuple(tokenizer_output["reconstructed"].shape),
                        )

                    if debug_overfit_single_batch:
                        snapshot = {
                            "step": global_step,
                            "reconstruction_loss": tokenizer_output["reconstruction_loss"].item(),
                            "cosine_similarity": tokenizer_output["cosine_similarity"].item(),
                        }
                        if debug_metrics["initial"] is None:
                            debug_metrics["initial"] = snapshot
                        debug_metrics["final"] = snapshot

                optimizer.zero_grad()
                total_loss.backward()
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

                    if tokenizer_output is not None:
                        writer.add_scalar("continuous_tokenizer/reconstruction_loss",
                                           tokenizer_output["reconstruction_loss"].item(), global_step)
                        writer.add_scalar("continuous_tokenizer/relative_l2_error",
                                           tokenizer_output["relative_l2_error"].item(), global_step)
                        writer.add_scalar("continuous_tokenizer/cosine_similarity",
                                           tokenizer_output["cosine_similarity"].item(), global_step)
                        writer.add_scalar("continuous_tokenizer/token_mean",
                                           tokenizer_output["tokens"].mean().item(), global_step)
                        writer.add_scalar("continuous_tokenizer/token_std",
                                           tokenizer_output["tokens"].std().item(), global_step)
                        writer.add_scalar("continuous_tokenizer/reconstruction_mean",
                                           tokenizer_output["reconstructed"].mean().item(), global_step)
                        writer.add_scalar("continuous_tokenizer/reconstruction_std",
                                           tokenizer_output["reconstructed"].std().item(), global_step)
                        LOGGER.info(
                            "  continuous_tokenizer | recon_loss=%.5f  rel_l2=%.5f  cos_sim=%.5f",
                            tokenizer_output["reconstruction_loss"].item(),
                            tokenizer_output["relative_l2_error"].item(),
                            tokenizer_output["cosine_similarity"].item(),
                        )

                if global_step > 0 and global_step % int(cfg.save_every_steps) == 0:
                    sp = ckpt_dir / f"step_{global_step:08d}.pth"
                    torch.save(_trainable_state_dict(comps, visual_tokenizer), sp)
                    LOGGER.info("Saved: %s", sp)

                global_step += 1

                if debug_overfit_single_batch and global_step >= debug_overfit_steps:
                    _report_debug_overfit(debug_metrics, LOGGER)
                    return

    finally:
        fp = ckpt_dir / f"last_{global_step:08d}.pth"
        torch.save(_trainable_state_dict(comps, visual_tokenizer), fp)
        LOGGER.info("Final: %s", fp)
        writer.close()


def _report_debug_overfit(debug_metrics: dict, logger: logging.Logger) -> None:
    """섹션 12: single-batch overfit 결과 요약. reconstruction loss가 명확히
    감소하지 않으면(80% 미만 감소) 다음 단계(pose conditioning/VQ) 진행 전에
    원인을 분석하라고 경고한다."""
    initial, final = debug_metrics["initial"], debug_metrics["final"]
    if initial is None or final is None:
        logger.warning("debug_overfit_single_batch: no tokenizer metrics were recorded")
        return

    loss_reduction_ratio = 1.0 - (final["reconstruction_loss"] / max(initial["reconstruction_loss"], 1e-12))
    logger.info(
        "\n[Single-batch overfit report]\n"
        "%-24s %12s %12s\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "loss_reduction_ratio=%.2f%%",
        "metric", f"step={initial['step']}", f"step={final['step']}",
        "reconstruction_loss", initial["reconstruction_loss"], final["reconstruction_loss"],
        "cosine_similarity", initial["cosine_similarity"], final["cosine_similarity"],
        loss_reduction_ratio * 100.0,
    )
    if loss_reduction_ratio < 0.8:
        logger.warning(
            "reconstruction loss dropped by only %.1f%% (< 80%%). Do NOT proceed to "
            "pose conditioning/VQ until this is investigated (check learning rate, "
            "embed_dim vs patch_dim, or whether the captured feature is actually varying).",
            loss_reduction_ratio * 100.0,
        )


def _trainable_state_dict(comps, visual_tokenizer=None) -> dict:
    """Flat, prefixed state dict matching the layout MimicMotionModel.load_state_dict
    (mimicmotion/utils/loader.py) expects, e.g. 'unet.xxx', 'pose_net.xxx'. When a
    continuous tokenizer is enabled, its weights are saved too, under a
    'continuous_tokenizer.*' prefix; MimicMotionModel.load_state_dict(strict=False)
    simply ignores these extra keys, so the checkpoint stays loadable by the
    existing inference pipeline either way."""
    ckpt = {}
    ckpt.update({f"unet.{k}": v for k, v in comps.unet.state_dict().items()})
    ckpt.update({f"pose_net.{k}": v for k, v in comps.pose_net.state_dict().items()})
    if visual_tokenizer is not None:
        ckpt.update({f"continuous_tokenizer.{k}": v for k, v in visual_tokenizer.state_dict().items()})
    return ckpt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/train_config.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
