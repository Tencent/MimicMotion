"""
inference_with_adapter.py

동일한 reference image/pose sequence/seed로 (1) baseline과 (2) continuous
adapter residual injection을 적용한 결과를 각각 생성해 비교한다.

train_vq.py가 저장하는 체크포인트는 unet.*/pose_net.* 외에
continuous_tokenizer.*/pose_bundle_encoder.*/pose_conditioned_film.*/
continuous_adapter_injector.* 를 추가로 포함한다. mimicmotion.utils.loader.
create_pipeline()은 이 extra key들을 (strict=False로) 무시하므로, 기본
파이프라인(baseline)은 checkpoint만으로 정상 로드된다. Continuous injection을
실제로 재현하려면 그 extra weight들로 별도 모듈을 재구성하고,
ContinuousAdapterInjector를 로드된 pipeline.unet에 다시 연결해야 한다 (학습
때와 동일한 hook 기반 메커니즘 -- continuous_adapter_injector.py 참고).

pipeline_mimicmotion.py의 실제 UNet 호출부(직접 읽어서 확인, 추측 아님)는
매 timestep마다 tile별로, CFG negative/positive 두 branch를 각각
batch_size=1로 따로 호출하며(`self.unet(latent_model_input[:1, idx], ...)`,
`self.unet(latent_model_input[1:, idx], ...)`), num_frames(=len(idx))도 tile마다
달라진다 -- 학습 때의 고정 (B, T)와 다르다. ContinuousAdapterInjector는 이걸
감안해 (batch_size, num_frames)를 매 unet(...) 호출 직전에 `sample.shape[:2]`
에서 자동으로 읽어오는 forward pre-hook을 쓰므로, 별도 수정 없이 이
파이프라인에도 그대로 붙는다.

실행:
    /opt/conda/envs/mimicmotion/bin/python inference_with_adapter.py \
        --config configs/inference_adapter.yaml
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dataset import MimicMotionFramesDataset, collate_fn
from inference import run_pipeline, device
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from visual_tokenizer import SpatioTemporalContinuousTokenizer, UNetFeatureCapture
from pose_bundle_encoder import PoseBundleEncoder
from pose_conditioned_film import PoseConditionedFiLM
from continuous_adapter_injector import ContinuousAdapterInjector


def _extract_prefixed(state_dict: dict, prefix: str) -> dict:
    p = prefix + "."
    return {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p)}


def _build_and_load_adapter(checkpoint: dict, train_cfg, unet, pose_net, device):
    """train_vq.py와 동일한 하이퍼파라미터로 tokenizer/pose_bundle_encoder/
    pose_conditioned_film/ContinuousAdapterInjector를 재구성하고, checkpoint의
    continuous_tokenizer.*/pose_bundle_encoder.*/pose_conditioned_film.*/
    continuous_adapter_injector.* 서브 state dict를 각각 로드한다."""
    visual_tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=unet.config.block_out_channels[1],
        embed_dim=int(train_cfg.get("visual_embed_dim", 64)),
        temporal_patch_size=int(train_cfg.get("temporal_patch_size", 4)),
        spatial_patch_height=int(train_cfg.get("spatial_patch_height", 4)),
        spatial_patch_width=int(train_cfg.get("spatial_patch_width", 3)),
    ).to(device)
    tok_sd = _extract_prefixed(checkpoint, "continuous_tokenizer")
    msg = visual_tokenizer.load_state_dict(tok_sd, strict=True)
    print(f"[adapter] loaded continuous_tokenizer: missing={msg.missing_keys} unexpected={msg.unexpected_keys}")

    pose_bundle_encoder = PoseBundleEncoder(
        pose_channels=int(train_cfg.pose_channels),
        pose_embed_dim=int(train_cfg.get("pose_embed_dim", 32)),
        hidden_dim=int(train_cfg.get("pose_hidden_dim", 128)),
        temporal_patch_size=int(train_cfg.get("temporal_patch_size", 4)),
        spatial_patch_height=int(train_cfg.get("spatial_patch_height", 4)),
        spatial_patch_width=int(train_cfg.get("spatial_patch_width", 3)),
        pose_normalization=str(train_cfg.get("pose_normalization", "none")),
        occupancy_activation_mode=str(train_cfg.get("occupancy_activation_mode", "abs_mean")),
        occupancy_normalization=str(train_cfg.get("occupancy_normalization", "sigmoid")),
        occupancy_sigmoid_temperature=float(train_cfg.get("occupancy_sigmoid_temperature", 0.01)),
        occupancy_active_threshold=float(train_cfg.get("occupancy_active_threshold", 0.6)),
        occupancy_background_threshold=float(train_cfg.get("occupancy_background_threshold", 0.4)),
    ).to(device)
    pose_sd = _extract_prefixed(checkpoint, "pose_bundle_encoder")
    msg = pose_bundle_encoder.load_state_dict(pose_sd, strict=True)
    print(f"[adapter] loaded pose_bundle_encoder: missing={msg.missing_keys} unexpected={msg.unexpected_keys}")

    pose_conditioned_film = PoseConditionedFiLM(
        visual_embed_dim=int(train_cfg.get("visual_embed_dim", 64)),
        pose_embed_dim=int(train_cfg.get("pose_embed_dim", 32)),
        hidden_dim=int(train_cfg.get("film_hidden_dim", 128)),
    ).to(device)
    film_sd = _extract_prefixed(checkpoint, "pose_conditioned_film")
    msg = pose_conditioned_film.load_state_dict(film_sd, strict=True)
    print(f"[adapter] loaded pose_conditioned_film: missing={msg.missing_keys} unexpected={msg.unexpected_keys}")

    pose_feature_capture = UNetFeatureCapture(pose_net.conv_layers)
    adapter_injector = ContinuousAdapterInjector(
        unet=unet,
        down_block=unet.down_blocks[1],
        visual_tokenizer=visual_tokenizer,
        pose_bundle_encoder=pose_bundle_encoder,
        pose_conditioned_film=pose_conditioned_film,
        pose_feature_capture=pose_feature_capture,
        injection_mode="none",  # start disabled; the caller flips this to "residual" when ready
        injection_scale_init=0.0,
        detach_tokenizer_input=bool(train_cfg.get("detach_tokenizer_input", True)),
        detach_pose_encoder_input=bool(train_cfg.get("detach_pose_encoder_input", True)),
    ).to(device)
    inj_sd = _extract_prefixed(checkpoint, "continuous_adapter_injector")
    msg = adapter_injector.load_state_dict(inj_sd, strict=True)
    print(f"[adapter] loaded continuous_adapter_injector: missing={msg.missing_keys} unexpected={msg.unexpected_keys} "
          f"(injection_scale={adapter_injector.injection_scale.item():.6f})")

    return adapter_injector


@torch.no_grad()
def main(args):
    cfg = OmegaConf.load(args.config)
    torch.set_default_dtype(torch.float16)

    ds = MimicMotionFramesDataset(
        str(cfg.train_manifest), int(cfg.resolution), int(cfg.num_frames),
    )
    sample_index = int(cfg.get("sample_index", 0))
    batch = collate_fn([ds[sample_index]])

    ref_image = batch["ref_image"][0]
    ref_pose = batch["ref_pose"][0]
    pose_images = batch["pose_images"][0]

    image_pixels = ref_image.unsqueeze(0).to(torch.float16)
    pose_pixels = torch.cat([ref_pose.unsqueeze(0), pose_images], dim=0).to(torch.float16)

    infer_config = OmegaConf.create({
        "base_model_path": cfg.base_model_path,
        "ckpt_path": cfg.ckpt_path,
    })
    pipeline = create_pipeline(infer_config, device)

    seed = int(cfg.get("seed", 42))
    task_config = OmegaConf.create({
        "seed": seed,
        "num_frames": int(cfg.get("tile_size", pose_pixels.size(0))),
        "frames_overlap": int(cfg.get("frames_overlap", 0)),
        "noise_aug_strength": float(cfg.get("noise_aug_strength", 0)),
        "num_inference_steps": int(cfg.get("num_inference_steps", 25)),
        "guidance_scale": float(cfg.get("guidance_scale", 2.0)),
    })

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = Path(cfg.ckpt_path).stem

    # ---- 1. baseline: plain pipeline, no adapter hook attached at all ----
    video_frames_baseline = run_pipeline(pipeline, image_pixels, pose_pixels, device, task_config)
    baseline_path = output_dir / f"sample{sample_index}_seed{seed}_baseline_{ckpt_tag}.mp4"
    save_to_mp4(video_frames_baseline, str(baseline_path), fps=int(cfg.get("fps", 15)))
    print(f"Saved baseline: {baseline_path}")

    # ---- 2. continuous adapter residual injection, using the checkpoint's ----
    #         own trained injection_scale (not reset to 0)
    train_cfg = OmegaConf.load(str(cfg.adapter_train_config))
    checkpoint = torch.load(str(cfg.ckpt_path), map_location="cpu", weights_only=True)
    adapter_injector = _build_and_load_adapter(checkpoint, train_cfg, pipeline.unet, pipeline.pose_net, device)
    adapter_injector.injection_mode = "residual"
    injection_scale_value = adapter_injector.injection_scale.item()

    video_frames_injected = run_pipeline(pipeline, image_pixels, pose_pixels, device, task_config)
    injected_path = output_dir / (
        f"sample{sample_index}_seed{seed}_continuous_injection_"
        f"scale{injection_scale_value:.4f}_{ckpt_tag}.mp4"
    )
    save_to_mp4(video_frames_injected, str(injected_path), fps=int(cfg.get("fps", 15)))
    print(f"Saved continuous injection (injection_scale={injection_scale_value:.6f}): {injected_path}")
    adapter_injector.remove()

    # ---- 3. optional side-by-side (left=baseline, right=continuous injection) ----
    if bool(cfg.get("save_side_by_side", True)):
        sbs_path = output_dir / f"sample{sample_index}_seed{seed}_sidebyside_{ckpt_tag}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(baseline_path), "-i", str(injected_path),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]",
            str(sbs_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Saved side-by-side (left=baseline, right=continuous_injection): {sbs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference_adapter.yaml")
    args = parser.parse_args()
    main(args)
