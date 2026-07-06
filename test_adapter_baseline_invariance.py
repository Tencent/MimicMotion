"""
test_adapter_baseline_invariance.py

실제 프로젝트 컴포넌트(UNet/PoseNet/VAE/CLIP, configs/train_config.yaml의
train_manifest에서 가져온 실제 샘플 1개)로 다음 세 설정에서 UNet
output/diffusion loss가 수치적으로 동일한지 검증한다.

A. adapter 완전 비활성화 (down_blocks[1]에 hook이 전혀 없음)
B. adapter 활성화 + adapter_injection_mode="none"
C. adapter 활성화 + adapter_injection_mode="residual" + injection_scale=0

A를 두 번 실행해 GPU 커널 자체의 부동소수점 비결정성 바닥값(A vs A2)을 먼저
측정한 뒤, B/C를 그 바닥값과 비교 가능한 tolerance로 판정한다 (추측 대신
실측).

이 테스트가 실패하면 다음 단계로 진행하지 않는다.

실행:
    /opt/conda/envs/mimicmotion/bin/python test_adapter_baseline_invariance.py
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import MimicMotionFramesDataset, collate_fn
from visual_tokenizer import SpatioTemporalContinuousTokenizer, UNetFeatureCapture
from pose_bundle_encoder import PoseBundleEncoder
from pose_conditioned_film import PoseConditionedFiLM
from continuous_adapter_injector import ContinuousAdapterInjector
from utils import (
    load_components,
    set_seed,
    encode_video_latents,
    encode_reference_image_embeds,
    encode_reference_image_latents,
    encode_pose_latents,
    get_added_time_ids,
)

CFG_PATH = "configs/train_config.yaml"


def _load_fixed_batch(cfg):
    ds = MimicMotionFramesDataset(str(cfg.train_manifest), int(cfg.resolution), int(cfg.num_frames))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=collate_fn)
    return next(iter(loader))


@torch.no_grad()
def _run_forward(comps, cfg, batch, device, adapter_injector=None):
    pixel_values = batch["pixel_values"].to(device)
    pose_images = batch["pose_images"].to(device)
    ref_image = batch["ref_image"].to(device)
    b, f = pixel_values.shape[:2]
    unet_dtype = next(comps.unet.parameters()).dtype
    sigma_data = float(cfg.get("sigma_data", 1.0))

    video_latents = encode_video_latents(comps, pixel_values)
    ref_embeds = encode_reference_image_embeds(comps, ref_image)

    torch.manual_seed(123)
    sigma = torch.full([b, 1, 1, 1, 1], 1.0, device=device)
    noise = torch.randn_like(video_latents)

    ref_cond_latents = encode_reference_image_latents(
        comps, ref_image, noise_aug_strength=float(cfg.noise_aug_strength), noise_for_aug=noise,
    )
    ref_image_latents = ref_cond_latents.unsqueeze(1).repeat(1, f, 1, 1, 1)

    sigma_sq = sigma ** 2
    c_in = 1.0 / (sigma_sq + sigma_data ** 2) ** 0.5
    c_noise = sigma.log() / 4.0
    noisy_latents = video_latents + noise * sigma
    input_latents = c_in * noisy_latents
    latent_input = torch.cat([input_latents, ref_image_latents], dim=2).to(unet_dtype)
    added_time_ids = get_added_time_ids(
        int(cfg.fps) - 1, int(cfg.motion_bucket_id), float(cfg.noise_aug_strength), b, device, unet_dtype,
    )
    c_noise_unet = c_noise.reshape(b).to(unet_dtype)

    pose_latents = encode_pose_latents(comps, pose_images)

    if adapter_injector is not None:
        adapter_injector.set_batch_context(batch_size=b, num_frames=f)

    model_pred = comps.unet(
        latent_input, c_noise_unet,
        encoder_hidden_states=ref_embeds.to(unet_dtype),
        added_time_ids=added_time_ids,
        pose_latents=pose_latents.to(unet_dtype),
        image_only_indicator=False,
    ).sample.float()

    c_skip = sigma_data ** 2 / (sigma_sq + sigma_data ** 2)
    c_out = -sigma * sigma_data / (sigma_sq + sigma_data ** 2) ** 0.5
    loss_weight = (sigma_sq + sigma_data ** 2) / (sigma * sigma_data) ** 2
    denoised = model_pred * c_out + c_skip * noisy_latents
    loss = (loss_weight * (denoised - video_latents) ** 2).mean()

    return model_pred, loss.item()


def _diff(a, b):
    return (a - b).abs().max().item(), (a - b).abs().mean().item()


def main():
    cfg = OmegaConf.load(CFG_PATH)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_ckpt = str(cfg.init_checkpoint) if cfg.get("init_checkpoint") else None
    comps = load_components(str(cfg.base_model_path), device, init_checkpoint=init_ckpt)
    comps.vae.requires_grad_(False); comps.vae.eval()
    comps.image_encoder.requires_grad_(False); comps.image_encoder.eval()
    comps.pose_net.requires_grad_(False); comps.pose_net.eval()
    comps.unet.eval()

    batch = _load_fixed_batch(cfg)

    visual_tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=comps.unet.config.block_out_channels[1],
        embed_dim=int(cfg.get("visual_embed_dim", 64)),
        temporal_patch_size=int(cfg.get("temporal_patch_size", 4)),
        spatial_patch_height=int(cfg.get("spatial_patch_height", 4)),
        spatial_patch_width=int(cfg.get("spatial_patch_width", 3)),
    ).to(device)
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
        occupancy_sigmoid_temperature=float(cfg.get("occupancy_sigmoid_temperature", 0.01)),
    ).to(device)
    pose_conditioned_film = PoseConditionedFiLM(
        visual_embed_dim=int(cfg.get("visual_embed_dim", 64)),
        pose_embed_dim=int(cfg.get("pose_embed_dim", 32)),
        hidden_dim=int(cfg.get("film_hidden_dim", 128)),
    ).to(device)
    pose_feature_capture = UNetFeatureCapture(comps.pose_net.conv_layers)

    # A (run twice, to measure the GPU kernel non-determinism floor)
    model_pred_A, loss_A = _run_forward(comps, cfg, batch, device, adapter_injector=None)
    model_pred_A2, loss_A2 = _run_forward(comps, cfg, batch, device, adapter_injector=None)
    aa_max, aa_mean = _diff(model_pred_A, model_pred_A2)
    print(f"[A vs A2 self-consistency] max_abs_diff={aa_max:.3e}  mean_abs_diff={aa_mean:.3e}  "
          f"loss_diff={abs(loss_A - loss_A2):.3e}")

    # B: adapter enabled, mode=none
    injector_B = ContinuousAdapterInjector(
        unet=comps.unet, down_block=comps.unet.down_blocks[1], visual_tokenizer=visual_tokenizer,
        pose_bundle_encoder=pose_bundle_encoder, pose_conditioned_film=pose_conditioned_film,
        pose_feature_capture=pose_feature_capture, injection_mode="none", injection_scale_init=0.0,
    ).to(device)
    model_pred_B, loss_B = _run_forward(comps, cfg, batch, device, adapter_injector=injector_B)
    injector_B.remove()

    # C: adapter enabled, mode=residual, injection_scale=0
    injector_C = ContinuousAdapterInjector(
        unet=comps.unet, down_block=comps.unet.down_blocks[1], visual_tokenizer=visual_tokenizer,
        pose_bundle_encoder=pose_bundle_encoder, pose_conditioned_film=pose_conditioned_film,
        pose_feature_capture=pose_feature_capture, injection_mode="residual", injection_scale_init=0.0,
    ).to(device)
    model_pred_C, loss_C = _run_forward(comps, cfg, batch, device, adapter_injector=injector_C)
    injector_C.remove()

    ab_max, ab_mean = _diff(model_pred_A, model_pred_B)
    ac_max, ac_mean = _diff(model_pred_A, model_pred_C)
    print(f"loss_A={loss_A:.10f}  loss_B={loss_B:.10f}  loss_C={loss_C:.10f}")
    print(f"[A vs B] max_abs_diff={ab_max:.3e}  mean_abs_diff={ab_mean:.3e}  loss_diff={abs(loss_A - loss_B):.3e}")
    print(f"[A vs C] max_abs_diff={ac_max:.3e}  mean_abs_diff={ac_mean:.3e}  loss_diff={abs(loss_A - loss_C):.3e}")

    # tolerance = max(measured GPU non-determinism floor, a small absolute floor) * safety margin
    tol = max(aa_max, 1e-5) * 10.0
    assert ab_max < tol, f"A vs B UNet output differs too much: {ab_max} (tolerance={tol})"
    assert ac_max < tol, f"A vs C UNet output differs too much: {ac_max} (tolerance={tol})"
    print(f"[OK] baseline invariance holds within measured GPU non-determinism floor (tol={tol:.3e})")


if __name__ == "__main__":
    main()
