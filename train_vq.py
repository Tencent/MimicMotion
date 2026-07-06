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
from pose_bundle_encoder import PoseBundleEncoder, pose_shuffle_diagnostic
from pose_conditioned_film import PoseConditionedFiLM, pose_film_shuffle_diagnostic
from continuous_adapter_injector import ContinuousAdapterInjector
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
    debug_overfit_pose_batch = bool(cfg.get("debug_overfit_pose_batch", False))
    debug_pose_overfit_steps = int(cfg.get("debug_pose_overfit_steps", 500))
    debug_overfit_film_batch = bool(cfg.get("debug_overfit_film_batch", False))
    debug_film_overfit_steps = int(cfg.get("debug_film_overfit_steps", 500))
    debug_overfit_injection_batch = bool(cfg.get("debug_overfit_injection_batch", False))
    debug_injection_overfit_steps = int(cfg.get("debug_injection_overfit_steps", 500))

    overfit_single = (
        bool(cfg.get("overfit_single_sample", False))
        or debug_overfit_single_batch
        or debug_overfit_pose_batch
        or debug_overfit_film_batch
        or debug_overfit_injection_batch
    )
    fixed_batch = None
    if overfit_single:
        fixed_batch = next(iter(train_loader))
        LOGGER.info("Overfit mode: ON")
    if debug_overfit_single_batch:
        LOGGER.info("Debug single-batch overfit mode: ON (%d steps, then stop and report)",
                    debug_overfit_steps)
    if debug_overfit_pose_batch:
        LOGGER.info("Debug pose single-batch overfit mode: ON (%d steps, then stop and report)",
                    debug_pose_overfit_steps)
    if debug_overfit_film_batch:
        LOGGER.info("Debug FiLM single-batch overfit mode: ON (%d steps, then stop and report)",
                    debug_film_overfit_steps)
    if debug_overfit_injection_batch:
        LOGGER.info("Debug adapter-injection single-batch overfit mode: ON (%d steps, then stop and report)",
                    debug_injection_overfit_steps)

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
    enable_pose_bundle_encoder_flag = bool(cfg.get("enable_pose_bundle_encoder", False))
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
        feature_name = str(cfg.get("visual_feature_name", "down_blocks[1].output"))
        LOGGER.info("Continuous tokenizer enabled on feature=%s", feature_name)

    # the pose branch aligns to the visual feature's spatial resolution, so
    # the visual feature hook must exist whenever EITHER branch is enabled,
    # even if the tokenizer itself is off.
    if (enable_continuous_tokenizer or enable_pose_bundle_encoder_flag) and feature_capture is None:
        feature_capture = UNetFeatureCapture(comps.unet.down_blocks[1])

    # Pose bundle encoder (Global Pose Embedding + Local Pose Occupancy only;
    # no FiLM/modulation/UNet injection/VQ yet). Selected feature =
    # PoseNet.conv_layers output (128ch, before final_proj), per the
    # 2026-07-03 pose-branch inspection (see pose_bundle_encoder.py
    # docstring). Parallel to the visual branch only -- never combined with
    # visual_tokens or fed back into the UNet.
    #
    # temporal_patch_size/spatial_patch_height/spatial_patch_width are
    # intentionally the SAME config keys the visual tokenizer reads above,
    # so the two branches can never disagree on patch grid (no separate
    # pose_temporal_patch_size etc. keys exist).
    enable_pose_bundle_encoder = enable_pose_bundle_encoder_flag
    pose_bundle_encoder = None
    pose_feature_capture = None

    if enable_pose_bundle_encoder:
        pose_bundle_encoder = PoseBundleEncoder(
            pose_channels=int(cfg.pose_channels),
            pose_embed_dim=int(cfg.get("pose_embed_dim", 32)),
            hidden_dim=int(cfg.get("pose_hidden_dim", 128)),
            temporal_patch_size=int(cfg.get("temporal_patch_size", 4)),
            spatial_patch_height=int(cfg.get("spatial_patch_height", 4)),
            spatial_patch_width=int(cfg.get("spatial_patch_width", 3)),
            pose_normalization=str(cfg.get("pose_normalization", "none")),
            occupancy_activation_mode=str(cfg.get("occupancy_activation_mode", "abs_mean")),
            occupancy_normalization=str(cfg.get("occupancy_normalization", "sigmoid")),
            occupancy_sigmoid_temperature=float(cfg.get("occupancy_sigmoid_temperature", 1.0)),
            occupancy_active_threshold=float(cfg.get("occupancy_active_threshold", 0.6)),
            occupancy_background_threshold=float(cfg.get("occupancy_background_threshold", 0.4)),
        ).to(device)
        pose_feature_capture = UNetFeatureCapture(comps.pose_net.conv_layers)
        pose_feature_name = str(cfg.get("pose_feature_name", "pose_net.conv_layers.output"))
        LOGGER.info("Pose bundle encoder enabled on feature=%s", pose_feature_name)

    # Pose-conditioned FiLM (first stage that lets pose information actually
    # affect visual tokens). Requires BOTH the tokenizer and pose bundle
    # encoder, since it modulates visual_tokens using global_pose_embedding /
    # local_pose_occupancy. Still never touches the UNet: the modulated
    # reconstruction only replaces the auxiliary tokenizer reconstruction
    # loss target, model_pred above already ran on the untouched feature.
    enable_pose_conditioned_film = bool(cfg.get("enable_pose_conditioned_film", False))
    if enable_pose_conditioned_film and not (enable_continuous_tokenizer and enable_pose_bundle_encoder):
        raise ValueError(
            "enable_pose_conditioned_film=true requires both enable_continuous_tokenizer=true "
            "and enable_pose_bundle_encoder=true (FiLM modulates visual_tokens using "
            "global_pose_embedding/local_pose_occupancy from the pose branch)."
        )
    pose_conditioned_film = None
    film_freeze_steps = int(cfg.get("film_freeze_steps", 0))
    if enable_pose_conditioned_film:
        pose_conditioned_film = PoseConditionedFiLM(
            visual_embed_dim=int(cfg.get("visual_embed_dim", 64)),
            pose_embed_dim=int(cfg.get("pose_embed_dim", 32)),
            hidden_dim=int(cfg.get("film_hidden_dim", 128)),
            film_scale_init=float(cfg.get("film_scale_init", 0.0)),
            last_linear_init=str(cfg.get("film_last_linear_init", "zero")),
            last_linear_init_std=float(cfg.get("film_last_linear_init_std", 0.02)),
        ).to(device)
        if film_freeze_steps > 0:
            for p in pose_conditioned_film.parameters():
                p.requires_grad_(False)
            LOGGER.info(
                "PoseConditionedFiLM parameters frozen for the first %d steps "
                "(film_freeze_steps), then unfrozen automatically.", film_freeze_steps,
            )
        LOGGER.info("Pose-conditioned FiLM enabled (visual tokens now modulated by pose)")

    # Continuous adapter injection: first stage where reconstructed_feature is
    # fed back into the UNet's own forward pass (residual, gated by a
    # learnable injection_scale starting at 0). Requires the full pipeline
    # (tokenizer -> pose bundle encoder -> FiLM) since ContinuousAdapterInjector
    # runs all three INSIDE a forward hook on down_blocks[1] (see
    # continuous_adapter_injector.py docstring for why this must be a hook
    # rather than code in the training loop: the injection has to happen
    # DURING the same comps.unet(...) call that produces model_pred).
    enable_continuous_adapter_injection = bool(cfg.get("enable_continuous_adapter_injection", False))
    adapter_injection_mode = str(cfg.get("adapter_injection_mode", "none"))
    if adapter_injection_mode not in ("none", "residual"):
        raise ValueError(
            f"Unknown adapter_injection_mode={adapter_injection_mode!r}; expected 'none' or 'residual'"
        )
    if enable_continuous_adapter_injection and not (
        enable_continuous_tokenizer and enable_pose_bundle_encoder and enable_pose_conditioned_film
    ):
        raise ValueError(
            "enable_continuous_adapter_injection=true requires enable_continuous_tokenizer=true, "
            "enable_pose_bundle_encoder=true, AND enable_pose_conditioned_film=true (the adapter "
            "pipeline is tokenizer -> pose bundle encoder -> FiLM -> reconstructed_feature, all of "
            "which now run inside a single forward hook)."
        )
    adapter_injector = None
    if enable_continuous_adapter_injection:
        adapter_injector = ContinuousAdapterInjector(
            unet=comps.unet,
            down_block=comps.unet.down_blocks[1],
            visual_tokenizer=visual_tokenizer,
            pose_bundle_encoder=pose_bundle_encoder,
            pose_conditioned_film=pose_conditioned_film,
            pose_feature_capture=pose_feature_capture,
            injection_mode=adapter_injection_mode,
            injection_scale_init=float(cfg.get("injection_scale_init", 0.0)),
            detach_tokenizer_input=bool(cfg.get("detach_tokenizer_input", True)),
            detach_pose_encoder_input=bool(cfg.get("detach_pose_encoder_input", True)),
        ).to(device)
        LOGGER.info(
            "Continuous adapter injection enabled (mode=%s) -- tokenizer/pose/FiLM now run "
            "inside a down_blocks[1] forward hook instead of the training loop.",
            adapter_injection_mode,
        )

    # baseline's own trainable-parameter setup (unet) is left exactly as-is;
    # the tokenizer/pose/film/injector branches are only ever ADDED as their
    # own optimizer param groups.
    param_groups = [{"params": list(comps.unet.parameters()), "lr": float(cfg.get("learning_rate", 1e-5))}]
    if enable_continuous_tokenizer:
        tokenizer_lr = float(cfg.tokenizer_learning_rate)
        param_groups.append({"params": list(visual_tokenizer.parameters()), "lr": tokenizer_lr})
    if enable_pose_bundle_encoder:
        pose_lr = float(cfg.pose_encoder_learning_rate)
        param_groups.append({"params": list(pose_bundle_encoder.parameters()), "lr": pose_lr})
    if enable_pose_conditioned_film:
        film_lr = float(cfg.film_learning_rate)
        param_groups.append({"params": list(pose_conditioned_film.parameters()), "lr": film_lr})
    if enable_continuous_adapter_injection:
        injection_lr = float(cfg.injection_learning_rate)
        param_groups.append({"params": [adapter_injector.injection_scale], "lr": injection_lr})

    trainable_params = [p for group in param_groups for p in group["params"]]
    total_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    tokenizer_param_count = sum(p.numel() for p in visual_tokenizer.parameters()) if enable_continuous_tokenizer else 0
    pose_param_count = sum(p.numel() for p in pose_bundle_encoder.parameters()) if enable_pose_bundle_encoder else 0
    film_param_count = sum(p.numel() for p in pose_conditioned_film.parameters()) if enable_pose_conditioned_film else 0
    injection_param_count = 1 if enable_continuous_adapter_injection else 0
    LOGGER.info(
        "Trainable params: %d (%.2f M)  [unet=%d, tokenizer=%d, pose_bundle=%d, film=%d, injection_scale=%d]",
        total_params, total_params / 1e6,
        sum(p.numel() for p in comps.unet.parameters()), tokenizer_param_count,
        pose_param_count, film_param_count, injection_param_count,
    )

    optimizer = AdamW(param_groups,
                      betas=(0.9, 0.999), weight_decay=1e-2)
    warmup = int(cfg.get("lr_warmup_steps", 500))
    # lr_decay_steps=0 (default) preserves the old behavior exactly: warmup
    # then flat at 1.0x forever. When >0, cosine-decays from 1.0x down to
    # lr_decay_min_scale over lr_decay_steps steps after warmup, then holds
    # at the floor.
    lr_decay_steps = int(cfg.get("lr_decay_steps", 0))
    lr_decay_min_scale = float(cfg.get("lr_decay_min_scale", 0.1))

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if lr_decay_steps <= 0:
            return 1.0
        progress = min(1.0, (step - warmup) / lr_decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_decay_min_scale + (1.0 - lr_decay_min_scale) * cosine

    lr_scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)

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
    pose_debug_metrics = {"initial": None, "final": None}
    film_debug_metrics = {"initial": None, "final": None}
    injection_debug_metrics = {"initial": None, "final": None}

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

                # pose_net runs with grad since it is being trained. This must
                # happen BEFORE comps.unet(...) below so that, if adapter
                # injection is enabled, pose_feature_capture.output is already
                # populated by the time the down_blocks[1] hook fires DURING
                # the unet(...) call (see continuous_adapter_injector.py).
                pose_latents = encode_pose_latents(comps, pose_images)

                if enable_continuous_adapter_injection:
                    adapter_injector.set_batch_context(batch_size=b, num_frames=f)

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

                total_loss = loss
                tokenizer_output = None
                pose_output = None
                film_output = None
                feature_5d = None
                visual_tokens = None

                if enable_continuous_adapter_injection:
                    # Everything (patchify/encode -> pose_bundle_encoder -> FiLM
                    # -> decode/unpatchify, and -- when adapter_injection_mode
                    # =="residual" -- the actual residual injection into
                    # feature_for_unet) already ran INSIDE the down_blocks[1]
                    # forward hook during the comps.unet(...) call above (see
                    # continuous_adapter_injector.py). We only read the results
                    # back out here to build total_loss/logging, exactly the
                    # same auxiliary losses as before plus injection_reg_loss.
                    tokenizer_output = adapter_injector.tokenizer_output
                    pose_output = adapter_injector.pose_output
                    film_output = adapter_injector.film_output
                    feature_5d = adapter_injector.original_feature_5d
                    visual_tokens = tokenizer_output["tokens"] if tokenizer_output is not None else None
                    if tokenizer_output is None or pose_output is None or film_output is None:
                        raise RuntimeError(
                            "ContinuousAdapterInjector hook did not populate its outputs -- "
                            "check that down_blocks[1] actually ran during comps.unet(...)."
                        )

                    lambda_recon = float(cfg.get("lambda_tokenizer_reconstruction", 1.0))
                    total_loss = total_loss + lambda_recon * tokenizer_output["reconstruction_loss"]

                    lambda_pose = float(cfg.get("lambda_pose_reconstruction", 1.0))
                    total_loss = total_loss + lambda_pose * pose_output["pose_reconstruction_loss"]

                    lambda_film_reg = float(cfg.get("lambda_film_reg", 1e-4))
                    film_reg_loss = film_output["gamma"].pow(2).mean() + film_output["beta"].pow(2).mean()
                    total_loss = total_loss + lambda_film_reg * film_reg_loss
                    film_output["film_reg_loss"] = film_reg_loss

                    injection_reg_loss = adapter_injector.injection_scale.pow(2)
                    if adapter_injection_mode == "residual":
                        lambda_injection_reg = float(cfg.get("lambda_injection_reg", 1e-4))
                        total_loss = total_loss + lambda_injection_reg * injection_reg_loss

                    if debug_overfit_single_batch:
                        snapshot = {
                            "step": global_step,
                            "reconstruction_loss": tokenizer_output["reconstruction_loss"].item(),
                            "cosine_similarity": tokenizer_output["cosine_similarity"].item(),
                        }
                        if debug_metrics["initial"] is None:
                            debug_metrics["initial"] = snapshot
                        debug_metrics["final"] = snapshot

                    if debug_overfit_pose_batch:
                        pose_snapshot = {
                            "step": global_step,
                            "pose_reconstruction_loss": pose_output["pose_reconstruction_loss"].item(),
                            "global_pose_std": pose_output["global_pose_std"].item(),
                            "global_pose_temporal_variance": pose_output["global_pose_temporal_variance"].item(),
                        }
                        if pose_debug_metrics["initial"] is None:
                            pose_debug_metrics["initial"] = pose_snapshot
                        pose_debug_metrics["final"] = pose_snapshot

                    if debug_overfit_film_batch:
                        modulation_strength = (
                            (film_output["modulated_tokens"] - visual_tokens).float().norm()
                            / (visual_tokens.float().norm() + 1e-8)
                        ).item()
                        film_snapshot = {
                            "step": global_step,
                            "reconstruction_loss": tokenizer_output["reconstruction_loss"].item(),
                            "film_scale": pose_conditioned_film.film_scale.item(),
                            "gamma_mean": film_output["gamma"].abs().mean().item(),
                            "beta_mean": film_output["beta"].abs().mean().item(),
                            "modulation_strength": modulation_strength,
                        }
                        if film_debug_metrics["initial"] is None:
                            film_debug_metrics["initial"] = film_snapshot
                        film_debug_metrics["final"] = film_snapshot

                    if debug_overfit_injection_batch:
                        injection_strength = (
                            (adapter_injector.feature_for_unet - feature_5d).float().norm()
                            / (feature_5d.float().norm() + 1e-8)
                        ).item()
                        injection_snapshot = {
                            "step": global_step,
                            "diffusion_loss": loss.item(),
                            "reconstruction_loss": tokenizer_output["reconstruction_loss"].item(),
                            "injection_scale": adapter_injector.injection_scale.item(),
                            "injection_strength": injection_strength,
                        }
                        if injection_debug_metrics["initial"] is None:
                            injection_debug_metrics["initial"] = injection_snapshot
                        injection_debug_metrics["final"] = injection_snapshot

                else:
                    # Continuous tokenizer auxiliary branch. reconstructed feature is
                    # NOT fed back into the UNet (feature_for_unet stays the
                    # original tensor the UNet already produced); this is purely
                    # an additional loss term. Disabled reproduces the baseline
                    # loss exactly (total_loss is loss).
                    #
                    # patchify/encode happen here (before the pose branch) so that
                    # visual_tokens exists in time for (a) PoseBundleEncoder's
                    # optional Nt/Nh/Nw consistency check and (b) PoseConditionedFiLM,
                    # which needs visual_tokens + the pose branch's
                    # global_pose_embedding/local_pose_occupancy together. decode/
                    # unpatchify/loss are finished further below, after the FiLM
                    # branch decides whether to use raw or modulated tokens.
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
                        # tokenizer output is never fed back into the UNet here.
                        patches = visual_tokenizer.patchify(feature_5d)
                        visual_tokens = visual_tokenizer.encode(patches)

                    # Pose bundle encoder auxiliary branch: Global Pose Embedding +
                    # Local Pose Occupancy only. Computed in parallel with the
                    # visual branch; NEVER fed back into the UNet (pose_latents
                    # above already went into comps.unet(...) unchanged). Its
                    # outputs are also the only pose signal PoseConditionedFiLM is
                    # allowed to see (below).
                    if enable_pose_bundle_encoder:
                        raw_pose_feature = pose_feature_capture.output
                        if raw_pose_feature is None:
                            raise RuntimeError(
                                "pose_bundle_encoder: feature capture hook on "
                                "comps.pose_net.conv_layers produced no output; "
                                "encode_pose_latents() above did not run as expected."
                            )
                        pose_branch_input = raw_pose_feature
                        if bool(cfg.get("detach_pose_encoder_input", True)):
                            pose_branch_input = pose_branch_input.detach()

                        target_visual_feature = feature_capture.output
                        target_hw = (target_visual_feature.shape[-2], target_visual_feature.shape[-1])

                        pose_output = pose_bundle_encoder(
                            pose_branch_input, target_spatial_size=target_hw,
                            batch_size=b, num_frames=f, visual_tokens=visual_tokens,
                        )
                        lambda_pose = float(cfg.get("lambda_pose_reconstruction", 1.0))
                        total_loss = total_loss + lambda_pose * pose_output["pose_reconstruction_loss"]

                        if global_step == 0 and is_rank0():
                            LOGGER.info(
                                "[PoseBundleEncoder shapes] original_pose_feature=%s "
                                "pose_aligned=%s pose_temporal_bundles=%s global_pose_raw=%s "
                                "global_pose_embedding=%s pose_activation=%s local_pose_occupancy=%s",
                                tuple(raw_pose_feature.shape), tuple(pose_output["pose_aligned"].shape),
                                tuple(pose_output["pose_temporal_bundles"].shape),
                                tuple(pose_output["global_pose_raw"].shape),
                                tuple(pose_output["global_pose_embedding"].shape),
                                tuple(pose_output["pose_activation"].shape),
                                tuple(pose_output["local_pose_occupancy"].shape),
                            )

                        if debug_overfit_pose_batch:
                            pose_snapshot = {
                                "step": global_step,
                                "pose_reconstruction_loss": pose_output["pose_reconstruction_loss"].item(),
                                "global_pose_std": pose_output["global_pose_std"].item(),
                                "global_pose_temporal_variance": pose_output["global_pose_temporal_variance"].item(),
                            }
                            if pose_debug_metrics["initial"] is None:
                                pose_debug_metrics["initial"] = pose_snapshot
                            pose_debug_metrics["final"] = pose_snapshot

                    # Pose-conditioned FiLM: first stage where pose information is
                    # actually applied to visual tokens. gamma/beta come from
                    # global_pose_embedding; local_pose_occupancy gates where the
                    # modulation is allowed to act. Still never touches the UNet --
                    # only the tokenizer's own auxiliary reconstruction path uses
                    # modulated_tokens instead of raw visual_tokens below.
                    tokens_for_decode = visual_tokens
                    if enable_pose_conditioned_film:
                        if global_step == film_freeze_steps and film_freeze_steps > 0:
                            for p in pose_conditioned_film.parameters():
                                p.requires_grad_(True)
                            LOGGER.info("PoseConditionedFiLM parameters unfrozen at step %d", global_step)
                        film_output = pose_conditioned_film(
                            visual_tokens=visual_tokens,
                            global_pose_embedding=pose_output["global_pose_embedding"],
                            local_pose_occupancy=pose_output["local_pose_occupancy"],
                        )
                        tokens_for_decode = film_output["modulated_tokens"]

                        if global_step == 0 and is_rank0():
                            LOGGER.info(
                                "[PoseConditionedFiLM shapes] visual_tokens=%s "
                                "global_pose_embedding=%s local_pose_occupancy=%s gamma=%s beta=%s "
                                "modulated_tokens=%s",
                                tuple(visual_tokens.shape), tuple(pose_output["global_pose_embedding"].shape),
                                tuple(pose_output["local_pose_occupancy"].shape),
                                tuple(film_output["gamma"].shape), tuple(film_output["beta"].shape),
                                tuple(film_output["modulated_tokens"].shape),
                            )

                    # Finish the tokenizer forward pass: decode + unpatchify the
                    # (possibly FiLM-modulated) tokens and compute the same metrics
                    # SpatioTemporalContinuousTokenizer.forward() would (kept
                    # manual, rather than calling .forward() again, so FiLM can
                    # sit between encode() and decode()). When FiLM is disabled,
                    # tokens_for_decode is exactly visual_tokens, so this is
                    # numerically identical to calling visual_tokenizer(feature_5d)
                    # directly (validated in test_pose_conditioned_film.py's
                    # zero-init identity test).
                    if enable_continuous_tokenizer:
                        decoded_patches = visual_tokenizer.decode(tokens_for_decode)
                        reconstructed = visual_tokenizer.unpatchify(decoded_patches, original_shape=feature_5d.shape)
                        x32 = feature_5d.float()
                        rec32 = reconstructed.float()
                        reconstruction_loss = torch.nn.functional.smooth_l1_loss(rec32, x32)
                        relative_l2_error = torch.norm(rec32 - x32) / (torch.norm(x32) + 1e-8)
                        cosine_similarity = torch.nn.functional.cosine_similarity(rec32, x32, dim=2).mean()
                        tokenizer_output = {
                            "patches": patches,
                            "tokens": visual_tokens,
                            "decoded_patches": decoded_patches,
                            "reconstructed": reconstructed,
                            "reconstruction_loss": reconstruction_loss,
                            "relative_l2_error": relative_l2_error,
                            "cosine_similarity": cosine_similarity,
                        }
                        lambda_recon = float(cfg.get("lambda_tokenizer_reconstruction", 1.0))
                        total_loss = total_loss + lambda_recon * reconstruction_loss

                        if film_output is not None:
                            lambda_film_reg = float(cfg.get("lambda_film_reg", 1e-4))
                            film_reg_loss = film_output["gamma"].pow(2).mean() + film_output["beta"].pow(2).mean()
                            total_loss = total_loss + lambda_film_reg * film_reg_loss
                            film_output["film_reg_loss"] = film_reg_loss

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

                        if debug_overfit_film_batch and film_output is not None:
                            modulation_strength = (
                                (film_output["modulated_tokens"] - visual_tokens).float().norm()
                                / (visual_tokens.float().norm() + 1e-8)
                            ).item()
                            film_snapshot = {
                                "step": global_step,
                                "reconstruction_loss": tokenizer_output["reconstruction_loss"].item(),
                                "film_scale": pose_conditioned_film.film_scale.item(),
                                "gamma_mean": film_output["gamma"].abs().mean().item(),
                                "beta_mean": film_output["beta"].abs().mean().item(),
                                "modulation_strength": modulation_strength,
                            }
                            if film_debug_metrics["initial"] is None:
                                film_debug_metrics["initial"] = film_snapshot
                            film_debug_metrics["final"] = film_snapshot

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

                    if pose_output is not None:
                        for key in (
                            "global_pose_mean", "global_pose_std", "global_pose_norm",
                            "global_pose_channel_variance", "global_pose_temporal_variance",
                            "global_pose_batch_variance", "occupancy_mean", "occupancy_std",
                            "occupancy_min", "occupancy_max", "occupancy_active_ratio",
                            "occupancy_background_ratio", "occupancy_temporal_variance",
                            "occupancy_spatial_variance",
                        ):
                            writer.add_scalar(f"pose_bundle/{key}", pose_output[key].item(), global_step)
                        writer.add_scalar("pose_bundle/pose_reconstruction_loss",
                                           pose_output["pose_reconstruction_loss"].item(), global_step)
                        encoder_grad_norm = _grad_norm(pose_bundle_encoder.global_encoder.parameters())
                        head_grad_norm = _grad_norm(pose_bundle_encoder.reconstruction_head.parameters())
                        writer.add_scalar("pose_bundle/encoder_grad_norm", encoder_grad_norm, global_step)
                        writer.add_scalar("pose_bundle/reconstruction_head_grad_norm", head_grad_norm, global_step)
                        LOGGER.info(
                            "  pose_bundle | recon_loss=%.5f  global_std=%.5f  "
                            "global_temporal_var=%.5f  occ_mean=%.5f  occ_active=%.3f",
                            pose_output["pose_reconstruction_loss"].item(),
                            pose_output["global_pose_std"].item(),
                            pose_output["global_pose_temporal_variance"].item(),
                            pose_output["occupancy_mean"].item(),
                            pose_output["occupancy_active_ratio"].item(),
                        )

                    if film_output is not None:
                        modulation_strength = (
                            (film_output["modulated_tokens"] - visual_tokens).float().norm()
                            / (visual_tokens.float().norm() + 1e-8)
                        ).item()
                        writer.add_scalar("pose_film/film_scale",
                                           pose_conditioned_film.film_scale.item(), global_step)
                        writer.add_scalar("pose_film/gamma_mean",
                                           film_output["gamma"].mean().item(), global_step)
                        writer.add_scalar("pose_film/gamma_std",
                                           film_output["gamma"].std().item(), global_step)
                        writer.add_scalar("pose_film/beta_mean",
                                           film_output["beta"].mean().item(), global_step)
                        writer.add_scalar("pose_film/beta_std",
                                           film_output["beta"].std().item(), global_step)
                        writer.add_scalar("pose_film/film_delta_norm",
                                           film_output["film_delta"].float().norm().item(), global_step)
                        writer.add_scalar("pose_film/modulated_token_mean",
                                           film_output["modulated_tokens"].mean().item(), global_step)
                        writer.add_scalar("pose_film/modulated_token_std",
                                           film_output["modulated_tokens"].std().item(), global_step)
                        writer.add_scalar("pose_film/modulation_strength", modulation_strength, global_step)
                        writer.add_scalar("pose_film/film_reg_loss",
                                           film_output["film_reg_loss"].item(), global_step)

                        # Diagnostic only -- never part of total_loss. Shuffles the
                        # pose branch's temporal-bundle order and checks whether
                        # modulated_tokens/reconstruction actually change.
                        shuffle_result = pose_film_shuffle_diagnostic(
                            pose_conditioned_film, visual_tokens,
                            pose_output["global_pose_embedding"], pose_output["local_pose_occupancy"],
                            decode_fn=lambda t: visual_tokenizer.unpatchify(
                                visual_tokenizer.decode(t), feature_5d.shape),
                        )
                        writer.add_scalar("pose_film/shuffle_modulated_token_l1_diff",
                                           shuffle_result["shuffle_modulated_token_l1_diff"], global_step)
                        writer.add_scalar("pose_film/shuffle_reconstruction_l1_diff",
                                           shuffle_result["shuffle_reconstruction_l1_diff"], global_step)
                        LOGGER.info(
                            "  pose_film | film_scale=%.5f  gamma_mean=%.5f  beta_mean=%.5f  "
                            "modulation_strength=%.5f  shuffle_mod_l1=%.5f",
                            pose_conditioned_film.film_scale.item(),
                            film_output["gamma"].abs().mean().item(),
                            film_output["beta"].abs().mean().item(),
                            modulation_strength,
                            shuffle_result["shuffle_modulated_token_l1_diff"],
                        )

                    if enable_continuous_adapter_injection:
                        injection_strength = (
                            (adapter_injector.feature_for_unet - feature_5d).float().norm()
                            / (feature_5d.float().norm() + 1e-8)
                        ).item()
                        feature_delta_for_log = adapter_injector.feature_for_unet - feature_5d
                        writer.add_scalar("continuous_adapter/injection_scale",
                                           adapter_injector.injection_scale.item(), global_step)
                        writer.add_scalar("continuous_adapter/feature_delta_norm",
                                           feature_delta_for_log.float().norm().item(), global_step)
                        writer.add_scalar("continuous_adapter/feature_delta_mean",
                                           feature_delta_for_log.float().mean().item(), global_step)
                        writer.add_scalar("continuous_adapter/feature_delta_std",
                                           feature_delta_for_log.float().std().item(), global_step)
                        writer.add_scalar("continuous_adapter/injected_feature_norm",
                                           adapter_injector.feature_for_unet.float().norm().item(), global_step)
                        writer.add_scalar("continuous_adapter/injection_strength", injection_strength, global_step)
                        writer.add_scalar("continuous_adapter/injection_reg_loss",
                                           injection_reg_loss.item(), global_step)
                        LOGGER.info(
                            "  continuous_adapter | mode=%s  injection_scale=%.5f  "
                            "injection_strength=%.5f  feature_delta_norm=%.5f",
                            adapter_injection_mode, adapter_injector.injection_scale.item(),
                            injection_strength, feature_delta_for_log.float().norm().item(),
                        )

                if global_step > 0 and global_step % int(cfg.save_every_steps) == 0:
                    sp = ckpt_dir / f"step_{global_step:08d}.pth"
                    torch.save(
                        _trainable_state_dict(comps, visual_tokenizer, pose_bundle_encoder,
                                               pose_conditioned_film, adapter_injector),
                        sp,
                    )
                    LOGGER.info("Saved: %s", sp)

                global_step += 1

                if debug_overfit_single_batch and global_step >= debug_overfit_steps:
                    _report_debug_overfit(debug_metrics, LOGGER)
                    return

                if debug_overfit_pose_batch and global_step >= debug_pose_overfit_steps:
                    _report_debug_pose_overfit(pose_debug_metrics, LOGGER)
                    return

                if debug_overfit_film_batch and global_step >= debug_film_overfit_steps:
                    _report_debug_film_overfit(film_debug_metrics, LOGGER)
                    return

                if debug_overfit_injection_batch and global_step >= debug_injection_overfit_steps:
                    _report_debug_injection_overfit(injection_debug_metrics, LOGGER)
                    return

    finally:
        fp = ckpt_dir / f"last_{global_step:08d}.pth"
        torch.save(
            _trainable_state_dict(comps, visual_tokenizer, pose_bundle_encoder,
                                   pose_conditioned_film, adapter_injector),
            fp,
        )
        LOGGER.info("Final: %s", fp)
        writer.close()


def _grad_norm(params) -> float:
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm() for g in grads])).item()


def _report_debug_pose_overfit(pose_debug_metrics: dict, logger: logging.Logger) -> None:
    """섹션 22: pose branch single-batch overfit 결과 요약."""
    initial, final = pose_debug_metrics["initial"], pose_debug_metrics["final"]
    if initial is None or final is None:
        logger.warning("debug_overfit_pose_batch: no pose metrics were recorded")
        return

    loss_reduction_ratio = 1.0 - (
        final["pose_reconstruction_loss"] / max(initial["pose_reconstruction_loss"], 1e-12)
    )
    logger.info(
        "\n[Pose single-batch overfit report]\n"
        "%-28s %12s %12s\n"
        "%-28s %12.6f %12.6f\n"
        "%-28s %12.6f %12.6f\n"
        "%-28s %12.6f %12.6f\n"
        "loss_reduction_ratio=%.2f%%",
        "metric", f"step={initial['step']}", f"step={final['step']}",
        "pose_reconstruction_loss", initial["pose_reconstruction_loss"], final["pose_reconstruction_loss"],
        "global_pose_std", initial["global_pose_std"], final["global_pose_std"],
        "global_pose_temporal_variance",
        initial["global_pose_temporal_variance"], final["global_pose_temporal_variance"],
        loss_reduction_ratio * 100.0,
    )
    if loss_reduction_ratio < 0.8:
        logger.warning(
            "pose_reconstruction_loss dropped by only %.1f%% (< 80%%). Investigate before "
            "proceeding (learning rate, whether overfit_fixed_sigma is set -- pose branch "
            "input itself does not depend on sigma, but check the captured feature is "
            "actually varying/non-degenerate).",
            loss_reduction_ratio * 100.0,
        )
    if final["global_pose_std"] < 1e-6:
        logger.warning("global_pose_embedding std collapsed to ~0 by the end of overfit")
    if final["global_pose_temporal_variance"] < 1e-6:
        logger.warning("global_pose_temporal_variance collapsed to ~0 by the end of overfit "
                       "(embedding no longer varies across temporal bundles)")


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


def _report_debug_film_overfit(film_debug_metrics: dict, logger: logging.Logger) -> None:
    """섹션10: pose-conditioned FiLM single-batch overfit 결과 요약. film_scale/
    gamma/beta가 0 근처에 머물러 있으면(이중 zero-init의 예상된 결과일 수 있음 --
    pose_conditioned_film.py 모듈 docstring 참고) 다음 단계로 넘어가지 말고
    film_scale_init/film_last_linear_init 대안을 검토하라고 경고한다."""
    initial, final = film_debug_metrics["initial"], film_debug_metrics["final"]
    if initial is None or final is None:
        logger.warning("debug_overfit_film_batch: no FiLM metrics were recorded")
        return

    loss_reduction_ratio = 1.0 - (
        final["reconstruction_loss"] / max(initial["reconstruction_loss"], 1e-12)
    )
    logger.info(
        "\n[Pose-conditioned FiLM single-batch overfit report]\n"
        "%-24s %12s %12s\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "loss_reduction_ratio=%.2f%%",
        "metric", f"step={initial['step']}", f"step={final['step']}",
        "reconstruction_loss", initial["reconstruction_loss"], final["reconstruction_loss"],
        "film_scale", initial["film_scale"], final["film_scale"],
        "gamma_mean(abs)", initial["gamma_mean"], final["gamma_mean"],
        "modulation_strength", initial["modulation_strength"], final["modulation_strength"],
        loss_reduction_ratio * 100.0,
    )
    if loss_reduction_ratio < 0.8:
        logger.warning(
            "FiLM reconstruction loss dropped by only %.1f%% (< 80%%). Do NOT proceed until "
            "investigated (check film_learning_rate, or whether film_scale/gamma/beta are "
            "stuck at 0 -- see the next two warnings).",
            loss_reduction_ratio * 100.0,
        )
    if abs(final["film_scale"]) < 1e-6:
        logger.warning(
            "film_scale is still ~0 at the end of overfit. With the default "
            "film_scale_init=0.0 AND film_last_linear_init='zero', gradient into film_scale "
            "and film_mlp is exactly 0 (see pose_conditioned_film.py docstring) -- try "
            "film_scale_init: 1e-3 (keeps modulated_tokens==visual_tokens at step 0 but "
            "unblocks gradient) or film_last_linear_init: small_normal."
        )
    if final["gamma_mean"] < 1e-6 and final["beta_mean"] < 1e-6:
        logger.warning("gamma/beta are still ~0 at the end of overfit (see film_scale warning above).")
    if final["modulation_strength"] <= 0.0:
        logger.warning("modulation_strength did not rise above 0 -- FiLM is not modulating visual_tokens at all.")


def _report_debug_injection_overfit(injection_debug_metrics: dict, logger: logging.Logger) -> None:
    """섹션10: continuous adapter residual injection single-batch overfit 결과
    요약. injection_scale이 전혀 학습되지 않거나 injection_strength가 폭증하면
    다음 단계로 넘어가지 말고 원인을 분석하라고 경고한다."""
    initial, final = injection_debug_metrics["initial"], injection_debug_metrics["final"]
    if initial is None or final is None:
        logger.warning("debug_overfit_injection_batch: no injection metrics were recorded")
        return

    diffusion_loss_reduction_ratio = 1.0 - (final["diffusion_loss"] / max(initial["diffusion_loss"], 1e-12))
    recon_loss_reduction_ratio = 1.0 - (
        final["reconstruction_loss"] / max(initial["reconstruction_loss"], 1e-12)
    )
    logger.info(
        "\n[Continuous adapter injection single-batch overfit report]\n"
        "%-24s %12s %12s\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "%-24s %12.6f %12.6f\n"
        "diffusion_loss_reduction_ratio=%.2f%%  reconstruction_loss_reduction_ratio=%.2f%%",
        "metric", f"step={initial['step']}", f"step={final['step']}",
        "diffusion_loss", initial["diffusion_loss"], final["diffusion_loss"],
        "reconstruction_loss", initial["reconstruction_loss"], final["reconstruction_loss"],
        "injection_scale", initial["injection_scale"], final["injection_scale"],
        "injection_strength", initial["injection_strength"], final["injection_strength"],
        diffusion_loss_reduction_ratio * 100.0, recon_loss_reduction_ratio * 100.0,
    )
    if not math.isfinite(final["diffusion_loss"]):
        logger.warning("diffusion_loss is NaN/Inf at the end of overfit -- investigate immediately, "
                       "do not proceed.")
    if abs(final["injection_scale"]) < 1e-6:
        logger.warning(
            "injection_scale is still ~0 at the end of overfit -- residual injection is not "
            "learning. Check injection_learning_rate, or (as with FiLM) whether the gradient "
            "path is starved by an upstream zero-init (film_scale/gamma/beta all still ~0)."
        )
    if final["injection_strength"] > 1.0:
        logger.warning(
            "injection_strength=%.3f (> 1.0) -- the injected feature now deviates from the "
            "original by more than its own norm. Consider lowering injection_learning_rate or "
            "raising lambda_injection_reg before proceeding.",
            final["injection_strength"],
        )
    if recon_loss_reduction_ratio < -0.2:
        logger.warning(
            "reconstruction_loss got markedly WORSE (%.1f%% change) while diffusion_loss was "
            "optimized -- the diffusion objective may be pulling the adapter away from accurate "
            "reconstruction. Investigate before proceeding.",
            recon_loss_reduction_ratio * 100.0,
        )


def _trainable_state_dict(
    comps, visual_tokenizer=None, pose_bundle_encoder=None, pose_conditioned_film=None,
    adapter_injector=None,
) -> dict:
    """Flat, prefixed state dict matching the layout MimicMotionModel.load_state_dict
    (mimicmotion/utils/loader.py) expects, e.g. 'unet.xxx', 'pose_net.xxx'. When a
    continuous tokenizer / pose bundle encoder / pose-conditioned FiLM / continuous
    adapter injector is enabled, their weights are saved too, under
    'continuous_tokenizer.*' / 'pose_bundle_encoder.*' / 'pose_conditioned_film.*' /
    'continuous_adapter_injector.*' prefixes; MimicMotionModel.load_state_dict
    (strict=False) simply ignores these extra keys, so the checkpoint stays
    loadable by the existing inference pipeline either way."""
    ckpt = {}
    ckpt.update({f"unet.{k}": v for k, v in comps.unet.state_dict().items()})
    ckpt.update({f"pose_net.{k}": v for k, v in comps.pose_net.state_dict().items()})
    if visual_tokenizer is not None:
        ckpt.update({f"continuous_tokenizer.{k}": v for k, v in visual_tokenizer.state_dict().items()})
    if pose_bundle_encoder is not None:
        ckpt.update({f"pose_bundle_encoder.{k}": v for k, v in pose_bundle_encoder.state_dict().items()})
    if pose_conditioned_film is not None:
        ckpt.update({f"pose_conditioned_film.{k}": v for k, v in pose_conditioned_film.state_dict().items()})
    if adapter_injector is not None:
        ckpt.update({f"continuous_adapter_injector.{k}": v for k, v in adapter_injector.state_dict().items()})
    return ckpt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/train_config.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
