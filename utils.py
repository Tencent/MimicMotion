from __future__ import annotations

import functools
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)
from diffusers.schedulers import EulerDiscreteScheduler
from torch.utils.checkpoint import checkpoint
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

_MIMICMOTION_ROOT = Path(__file__).resolve().parent / "MimicMotion"
if str(_MIMICMOTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_MIMICMOTION_ROOT))

from mimicmotion.modules.pose_net import PoseNet
from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel

LOGGER = logging.getLogger(__name__)


# ── 기본 유틸리티 ─────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── 모델 컴포넌트 ─────────────────────────────────────────────────

@dataclass
class Components:
    vae: AutoencoderKLTemporalDecoder
    image_encoder: CLIPVisionModelWithProjection
    feature_extractor: CLIPImageProcessor
    unet: UNetSpatioTemporalConditionModel
    pose_net: PoseNet
    scheduler: EulerDiscreteScheduler
    vae_image_processor: VaeImageProcessor


def _enable_gradient_checkpointing(model: torch.nn.Module) -> int:
    gc_func = functools.partial(checkpoint, use_reentrant=False)
    count = 0
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            try:
                m.gradient_checkpointing = True
                m._gradient_checkpointing_func = gc_func
                count += 1
            except Exception:
                pass
    return count


def _load_checkpoint(
    init_checkpoint: str,
    unet: UNetSpatioTemporalConditionModel,
    pose_net: PoseNet,
) -> None:
    ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(ckpt)}")

    def _extract(prefix: str) -> Dict[str, Any]:
        return {k[len(prefix)+1:]: v for k, v in ckpt.items() if k.startswith(prefix + ".")}

    unet_sd = ckpt.get("unet") or _extract("unet")
    pose_sd = ckpt.get("pose_net") or _extract("pose_net")

    if unet_sd:
        msg = unet.load_state_dict(unet_sd, strict=False)
        LOGGER.info("Loaded UNet | missing=%d unexpected=%d",
                    len(msg.missing_keys), len(msg.unexpected_keys))
    else:
        LOGGER.warning("Checkpoint has no UNet weights")

    if pose_sd:
        msg = pose_net.load_state_dict(pose_sd, strict=False)
        LOGGER.info("Loaded PoseNet | missing=%d unexpected=%d",
                    len(msg.missing_keys), len(msg.unexpected_keys))
    else:
        LOGGER.warning("Checkpoint has no PoseNet weights")


def load_components(
    base_model_path: str,
    device: torch.device,
    init_checkpoint: Optional[str] = None,
) -> Components:
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        base_model_path, subfolder="vae",
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_model_path, subfolder="image_encoder",
        torch_dtype=torch.float16, variant="fp16",
    ).to(device)

    feature_extractor = CLIPImageProcessor.from_pretrained(
        base_model_path, subfolder="feature_extractor",
    )

    scheduler = EulerDiscreteScheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        base_model_path, subfolder="unet",
        torch_dtype=torch.float32,
    ).to(device)

    n = _enable_gradient_checkpointing(unet)
    LOGGER.info("Enabled gradient checkpointing on %d submodules", n)

    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])

    if init_checkpoint:
        _load_checkpoint(init_checkpoint, unet, pose_net)

    pose_net = pose_net.to(device)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    pose_net.requires_grad_(True)

    return Components(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        pose_net=pose_net,
        scheduler=scheduler,
        vae_image_processor=VaeImageProcessor(vae_scale_factor=8),
    )


# ── 인코딩 함수 ───────────────────────────────────────────────────

def encode_reference_image_embeds(
    comps: Components, ref_image: torch.Tensor
) -> torch.Tensor:
    image = (ref_image + 1.0) / 2.0
    image = _resize_with_antialiasing(image, (224, 224))
    image = comps.feature_extractor(
        images=image,
        do_normalize=True, do_center_crop=False,
        do_resize=False, do_rescale=False,
        return_tensors="pt",
    ).pixel_values.to(
        device=ref_image.device,
        dtype=next(comps.image_encoder.parameters()).dtype,
    )
    return comps.image_encoder(image).image_embeds.unsqueeze(1)


def encode_reference_image_latents(
    comps: Components,
    ref_image: torch.Tensor,
    noise_aug_strength: float,
    noise_for_aug: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    image_01 = (ref_image + 1.0) / 2.0
    image = comps.vae_image_processor.preprocess(
        image_01, height=ref_image.shape[-2], width=ref_image.shape[-1],
    ).to(device=ref_image.device, dtype=next(comps.vae.parameters()).dtype)
    latents = comps.vae.encode(image).latent_dist.mode()
    if noise_for_aug is not None:
        latents = noise_for_aug[:, 0] * noise_aug_strength + latents
    else:
        latents = latents + noise_aug_strength * torch.randn_like(latents)
    return latents


def encode_video_latents(
    comps: Components, pixel_values: torch.Tensor
) -> torch.Tensor:
    b, f, c, h, w = pixel_values.shape
    flat_01 = (pixel_values.reshape(b * f, c, h, w) + 1.0) / 2.0
    flat = comps.vae_image_processor.preprocess(flat_01, height=h, width=w).to(
        dtype=next(comps.vae.parameters()).dtype,
        device=pixel_values.device,
    )
    latents = comps.vae.encode(flat).latent_dist.sample() * comps.vae.config.scaling_factor
    return latents.view(b, f, -1, latents.shape[-2], latents.shape[-1]).contiguous()


def encode_pose_latents(
    comps: Components, pose_images: torch.Tensor
) -> torch.Tensor:
    b, f, c, h, w = pose_images.shape
    flat = pose_images.reshape(b * f, c, h, w).to(
        dtype=next(comps.pose_net.parameters()).dtype
    )
    return comps.pose_net(flat)


def get_added_time_ids(
    fps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    added = torch.tensor(
        [fps, motion_bucket_id, noise_aug_strength],
        device=device, dtype=dtype,
    )
    return added.unsqueeze(0).repeat(batch_size, 1)
