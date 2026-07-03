"""
visual_tokenizer.py

선택된 UNet feature([B,T,C,H,W])를 non-overlapping spatio-temporal patch로
나누고, linear encoder/decoder를 통해 continuous visual token으로 변환한 뒤
다시 원본 feature shape으로 정확히 복원하는 모듈.

이 단계에서는 VQ/codebook/FiLM/pose encoder를 구현하지 않는다. tokenizer는
UNet feature를 그대로 통과시키는 auxiliary branch이며(reconstructed feature를
UNet에 다시 주입하지 않음), 목적은 오직 patchify/encode/decode/unpatchify가
정확히 동작하고 bottleneck이 학습 가능한지 검증하는 것이다.

선택된 feature 위치 (2026-07-03 feature inspection 결과, 추측 아님 -
mimicmotion/modules/unet.py 실제 실행으로 확인):
    파일명: mimicmotion/modules/unet.py
    클래스명: UNetSpatioTemporalConditionModel
    함수명: forward (down_blocks 순회 루프, L450-465)
    feature가 생성되는 코드 위치:
        down_blocks[1](CrossAttnDownBlockSpatioTemporal)의 반환값 (sample, res_samples)
        중 sample. unet.py 내부에서는 `sample, res_samples = downsample_block(...)`.
    원본 feature shape (train_config.yaml: resolution=576, num_frames=8 기준 실측):
        (B*T, C, H, W) = (8, 640, 32, 18)   [B=1, T=8]
    각 dimension의 의미:
        dim0 = B*T (unet.py L434 `sample.flatten(0, 1)`로 생성됨. B가 바깥쪽,
               T가 안쪽 축으로 flatten되어 있음 = index는 b*T+t 순서. 따라서
               복원은 반드시 `.reshape(B, T, C, H, W)` — B가 먼저, T가 다음.
               (unet.py L502의 최종 출력 복원 코드 `sample.reshape(batch_size,
               num_frames, *sample.shape[1:])`와 동일한 규약.)
        dim1 = C = 640 (block_out_channels[1])
        dim2 = H = 32, dim3 = W = 18 (VAE latent 해상도의 1/4, down_blocks 2단계 통과)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


def is_rank0() -> bool:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


# --------------------------------------------------------------------------
# B*T <-> B,T 변환. mimicmotion/modules/unet.py의 실제 flatten/reshape 규약
# (L434 flatten(0,1), L502 reshape(batch_size, num_frames, ...))과 동일하게,
# B가 바깥쪽/T가 안쪽 축인 row-major 순서를 그대로 따른다.
# --------------------------------------------------------------------------

def restore_video_dimension(
    feature: torch.Tensor,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    """Convert [B*T, C, H, W] to [B, T, C, H, W].

    unet.py L434(`sample.flatten(0, 1)`)와 정확히 반대 방향 연산이다: 원본
    tensor가 [B, T, ...] 순서로 flatten되어 있다고 가정하고 그 순서 그대로
    reshape한다 (B가 바깥쪽, T가 안쪽 = index b*T + t).
    """
    if feature.dim() != 4:
        raise ValueError(
            "restore_video_dimension expects a 4D tensor [B*T, C, H, W], "
            f"got shape={tuple(feature.shape)} (dim={feature.dim()})"
        )
    bt, c, h, w = feature.shape
    if bt != batch_size * num_frames:
        raise ValueError(
            "feature.shape[0] does not match batch_size * num_frames: "
            f"feature.shape[0]={bt}, batch_size={batch_size}, num_frames={num_frames}, "
            f"batch_size*num_frames={batch_size * num_frames}, full shape={tuple(feature.shape)}"
        )
    return feature.reshape(batch_size, num_frames, c, h, w)


def flatten_video_dimension(feature: torch.Tensor) -> torch.Tensor:
    """Convert [B, T, C, H, W] to [B*T, C, H, W]. restore_video_dimension의 역함수."""
    if feature.dim() != 5:
        raise ValueError(
            "flatten_video_dimension expects a 5D tensor [B, T, C, H, W], "
            f"got shape={tuple(feature.shape)} (dim={feature.dim()})"
        )
    b, t, c, h, w = feature.shape
    return feature.reshape(b * t, c, h, w)


class UNetFeatureCapture:
    """UNet의 특정 서브모듈(예: down_blocks[1])의 forward 출력을 매 스텝 참조로만
    저장하는 상시(self-removing 아님) hook. FeatureInspector(1회성, metadata만)와
    달리 이건 실제 tensor를 (grad 그래프 포함) 들고 있어야 tokenizer 입력으로 쓸 수
    있다. 텐서를 clone/detach하지 않는다 — detach 여부는 호출부(config)가 결정한다.
    UNet forward의 출력/입력을 변경하지 않는다(hook은 항상 None을 반환).
    """

    def __init__(self, module: nn.Module):
        self.output: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs, output) -> None:
        # down_blocks[i]는 (sample, res_samples) 튜플을 반환한다 (unet.py L459-463).
        self.output = output[0] if isinstance(output, tuple) else output

    def remove(self) -> None:
        self._handle.remove()


class SpatioTemporalContinuousTokenizer(nn.Module):
    """Converts video UNet features into non-overlapping spatio-temporal bundle
    tokens and reconstructs them. VQ/codebook은 포함하지 않는 continuous
    bottleneck이다 (다음 단계에서 quantization을 추가하기 위한 준비 단계).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        temporal_patch_size: int,
        spatial_patch_height: int,
        spatial_patch_width: int,
        identity_init: bool = False,
        warn_large_patch_dim: bool = True,
        large_patch_dim_threshold: int = 100_000,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_height = spatial_patch_height
        self.spatial_patch_width = spatial_patch_width
        self.patch_dim = (
            temporal_patch_size * spatial_patch_height * spatial_patch_width * in_channels
        )

        if warn_large_patch_dim and self.patch_dim > large_patch_dim_threshold:
            warnings.warn(
                f"[SpatioTemporalContinuousTokenizer] patch_dim={self.patch_dim} is large "
                f"(> {large_patch_dim_threshold}). encoder/decoder Linear layers will have "
                f"~{self.patch_dim * embed_dim * 2:,} params. Consider reducing "
                "embed_dim, increasing spatial patch size, or using a factorized "
                "projection instead of a single Linear layer."
            )

        # 5. Projection encoder / decoder (단순 linear, 이번 단계에서는 conv/attention 미사용)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, embed_dim),
        )
        self.decoder = nn.Linear(embed_dim, self.patch_dim)
        nn.init.zeros_(self.decoder.bias)

        # 6. identity init (embed_dim == patch_dim일 때만 의미가 있음)
        if identity_init:
            if embed_dim != self.patch_dim:
                warnings.warn(
                    "identity_init=True but embed_dim "
                    f"({embed_dim}) != patch_dim ({self.patch_dim}); "
                    "identity 초기화를 건너뛰고 기본 PyTorch 초기화를 사용합니다."
                )
            else:
                nn.init.eye_(self.encoder[1].weight)
                nn.init.zeros_(self.encoder[1].bias)
                nn.init.eye_(self.decoder.weight)

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        if is_rank0():
            print(
                "[SpatioTemporalContinuousTokenizer] "
                f"patch_dim={self.patch_dim} "
                f"(Lt={temporal_patch_size} * Ph={spatial_patch_height} * "
                f"Pw={spatial_patch_width} * C={in_channels}), "
                f"embed_dim={embed_dim}, "
                f"encoder_params={encoder_params:,}, decoder_params={decoder_params:,}"
            )

    # ------------------------------------------------------------------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x: [B, T, C, H, W]
        Output: patches: [B, Nt, Nh, Nw, Lt * Ph * Pw * C]
        """
        if x.dim() != 5:
            raise ValueError(f"patchify expects 5D [B,T,C,H,W], got shape={tuple(x.shape)}")
        b, t, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"patchify: channel mismatch, expected in_channels={self.in_channels}, "
                f"got C={c} from shape={tuple(x.shape)}"
            )
        lt, ph, pw = self.temporal_patch_size, self.spatial_patch_height, self.spatial_patch_width
        if t % lt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                "patchify: input T,H,W must be divisible by patch sizes. "
                f"got T={t}, H={h}, W={w}; "
                f"temporal_patch_size={lt}, spatial_patch_height={ph}, spatial_patch_width={pw} "
                f"(T%Lt={t % lt}, H%Ph={h % ph}, W%Pw={w % pw})"
            )
        nt, nh, nw = t // lt, h // ph, w // pw

        x = x.contiguous()                                     # [B, T, C, H, W]
        x = x.reshape(b, nt, lt, c, nh, ph, nw, pw)             # [B, Nt, Lt, C, Nh, Ph, Nw, Pw]
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)                   # [B, Nt, Nh, Nw, Lt, Ph, Pw, C]
        x = x.contiguous()
        patches = x.reshape(b, nt, nh, nw, lt * ph * pw * c)    # [B, Nt, Nh, Nw, patch_dim]
        return patches

    def unpatchify(
        self,
        decoded_patches: torch.Tensor,
        original_shape: Tuple[int, int, int, int, int],
    ) -> torch.Tensor:
        """
        Input:  decoded_patches: [B, Nt, Nh, Nw, patch_dim], original_shape: (B, T, C, H, W)
        Output: reconstructed: [B, T, C, H, W]
        patchify()와 정확히 역순인 permute/reshape을 적용한다 (섹션 13 정확성 테스트로 검증됨).
        """
        b, t, c, h, w = original_shape
        lt, ph, pw = self.temporal_patch_size, self.spatial_patch_height, self.spatial_patch_width
        if t % lt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                "unpatchify: original_shape T,H,W must be divisible by patch sizes. "
                f"got T={t}, H={h}, W={w}; "
                f"temporal_patch_size={lt}, spatial_patch_height={ph}, spatial_patch_width={pw}"
            )
        nt, nh, nw = t // lt, h // ph, w // pw
        expected_shape = (b, nt, nh, nw, lt * ph * pw * c)
        if tuple(decoded_patches.shape) != expected_shape:
            raise ValueError(
                f"unpatchify: expected decoded_patches shape {expected_shape}, "
                f"got {tuple(decoded_patches.shape)}"
            )

        x = decoded_patches.reshape(b, nt, nh, nw, lt, ph, pw, c)  # [B,Nt,Nh,Nw,Lt,Ph,Pw,C]
        x = x.permute(0, 1, 4, 7, 2, 5, 3, 6)                      # [B,Nt,Lt,C,Nh,Ph,Nw,Pw]
        x = x.contiguous()
        reconstructed = x.reshape(b, t, c, h, w)                   # [B,T,C,H,W]
        return reconstructed

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Input:  patches: [B, Nt, Nh, Nw, patch_dim]
        Output: tokens: [B, Nt, Nh, Nw, embed_dim]
        """
        if patches.shape[-1] != self.patch_dim:
            raise ValueError(
                f"encode: expected last dim {self.patch_dim} (patch_dim), "
                f"got {patches.shape[-1]} from shape={tuple(patches.shape)}"
            )
        return self.encoder(patches)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Input:  tokens: [B, Nt, Nh, Nw, embed_dim]
        Output: decoded_patches: [B, Nt, Nh, Nw, patch_dim]
        """
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                f"decode: expected last dim {self.embed_dim} (embed_dim), "
                f"got {tokens.shape[-1]} from shape={tuple(tokens.shape)}"
            )
        return self.decoder(tokens)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, c, h, w = x.shape
        patches = self.patchify(x)
        tokens = self.encode(patches)
        decoded_patches = self.decode(tokens)
        reconstructed = self.unpatchify(decoded_patches, original_shape=(b, t, c, h, w))

        # loss/metric은 mixed precision(fp16/bf16) 입력이 들어와도 float32로 계산한다.
        x32 = x.float()
        rec32 = reconstructed.float()
        reconstruction_loss = F.smooth_l1_loss(rec32, x32)
        relative_l2_error = torch.norm(rec32 - x32) / (torch.norm(x32) + 1e-8)
        # channel(dim=2) 기준 cosine similarity를 계산한 뒤 전체(B,T,H,W) 평균.
        cosine_similarity = F.cosine_similarity(rec32, x32, dim=2).mean()

        return {
            "patches": patches,
            "tokens": tokens,
            "decoded_patches": decoded_patches,
            "reconstructed": reconstructed,
            "reconstruction_loss": reconstruction_loss,
            "relative_l2_error": relative_l2_error,
            "cosine_similarity": cosine_similarity,
        }
