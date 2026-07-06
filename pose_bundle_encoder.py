"""
pose_bundle_encoder.py

Pose feature(PoseNet.conv_layers 출력, 128ch)를 selected visual feature와
동일한 spatio-temporal bundle 구조로 정렬하고, 각 bundle에 대한
Global Pose Embedding과 Local Pose Occupancy만 계산하는 auxiliary branch.

이 단계에서는 FiLM/gamma/beta/visual token modulation/continuous
bottleneck/UNet 주입/VQ를 전혀 구현하지 않는다. Pose branch와 visual
branch는 순수하게 병렬로만 실행되며, 이 모듈의 출력을 visual_tokens에
곱하거나 더하는 어떤 연산도 없다.

선택된 pose feature 위치 (2026-07-03 실제 hook으로 확인, 추측 아님):
    파일명: mimicmotion/modules/pose_net.py
    클래스명: PoseNet
    feature 생성 위치: self.conv_layers(x) 의 반환값 (self.final_proj 이전)
    실측 shape (resolution=576, pose 타겟 1024x576 기준): (B*T, 128, 128, 72)
    각 dimension 의미:
        dim0 = B*T. utils.py encode_pose_latents()의
               `pose_images.reshape(b*f, c, h, w)`와 동일한 규약(B 바깥/T
               안쪽) -- unet.py L434 flatten(0,1)과 정확히 같은 순서이므로
               visual_tokenizer.restore_video_dimension을 그대로 재사용 가능.
        dim1 = Cp = 128 (final_proj 이전 채널 폭)
        dim2,3 = Hp,Wp = 128,72 (pose 입력 1024x576을 conv stride-2 3회로
                 1/8 다운샘플; final_proj은 1x1 conv라 해상도 불변)
    activation 특성: SiLU 이후 값 (min=-0.2785, max=9.3075, mean=0.0612 --
    continuous, unbounded above, discrete label/heatmap 아님) -> bilinear
    interpolation이 적절.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_tokenizer import is_rank0, restore_video_dimension


# ---------------------------------------------------------------------------
# 1. Pose feature spatial/temporal alignment
# ---------------------------------------------------------------------------

class PoseFeatureAligner(nn.Module):
    """Convert the project-specific pose feature into
    [B, T, Cp, Hx, Wx] aligned with the selected visual feature.

    Learnable parameter가 없다 (bilinear interpolation + optional
    normalization만 수행). normalization을 config로 선택할 때만
    channel_layernorm에 한해 학습 가능한 LayerNorm affine이 추가된다.
    """

    def __init__(self, pose_channels: int, normalization: str = "none") -> None:
        super().__init__()
        if normalization not in ("none", "channel_layernorm", "channel_standardize"):
            raise ValueError(
                f"Unknown pose_normalization={normalization!r}; "
                "expected one of 'none', 'channel_layernorm', 'channel_standardize'"
            )
        self.pose_channels = pose_channels
        self.normalization = normalization
        self.channel_ln = nn.LayerNorm(pose_channels) if normalization == "channel_layernorm" else None

    def _restore_5d(self, pose_feature: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        if pose_feature.dim() == 4:
            return restore_video_dimension(pose_feature, batch_size=batch_size, num_frames=num_frames)
        if pose_feature.dim() == 5:
            b, t, c, _h, _w = pose_feature.shape
            if b != batch_size or t != num_frames:
                raise ValueError(
                    "PoseFeatureAligner: pose_feature is already 5D but its (B, T) "
                    f"= ({b}, {t}) does not match the expected (batch_size, num_frames) "
                    f"= ({batch_size}, {num_frames})"
                )
            return pose_feature
        raise ValueError(
            "PoseFeatureAligner expects a 4D [B*T, Cp, Hp, Wp] or 5D [B, T, Cp, Hp, Wp] "
            f"tensor, got shape={tuple(pose_feature.shape)} (dim={pose_feature.dim()})"
        )

    def _normalize(self, pose_aligned: torch.Tensor) -> torch.Tensor:
        if self.normalization == "none":
            return pose_aligned
        if self.normalization == "channel_layernorm":
            # [B,T,C,H,W] -> [B,T,H,W,C] -> LayerNorm(C) -> back
            x = pose_aligned.permute(0, 1, 3, 4, 2)
            x = self.channel_ln(x)
            return x.permute(0, 1, 4, 2, 3)
        if self.normalization == "channel_standardize":
            mean = pose_aligned.mean(dim=(1, 3, 4), keepdim=True)
            std = pose_aligned.std(dim=(1, 3, 4), keepdim=True)
            return (pose_aligned - mean) / (std + 1e-6)
        raise AssertionError("unreachable")  # validated in __init__

    def forward(
        self,
        pose_feature: torch.Tensor,
        batch_size: int,
        num_frames: int,
        target_spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        pose_5d = self._restore_5d(pose_feature, batch_size, num_frames)
        b, t, c, h, w = pose_5d.shape
        if c != self.pose_channels:
            raise ValueError(
                f"PoseFeatureAligner: expected pose_channels={self.pose_channels}, "
                f"got C={c} from shape={tuple(pose_5d.shape)}"
            )
        hx, wx = target_spatial_size
        if (h, w) != (hx, wx):
            flat = pose_5d.reshape(b * t, c, h, w)
            flat = F.interpolate(flat, size=(hx, wx), mode="bilinear", align_corners=False)
            pose_5d = flat.reshape(b, t, c, hx, wx)
        return self._normalize(pose_5d)


# ---------------------------------------------------------------------------
# 2. Global Pose Embedding
# ---------------------------------------------------------------------------

class GlobalPoseEncoder(nn.Module):
    """Generate one global pose embedding for each temporal bundle."""

    def __init__(
        self,
        pose_channels: int,
        pose_embed_dim: int,
        hidden_dim: int,
        pooling_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.pose_channels = pose_channels
        self.pose_embed_dim = pose_embed_dim
        self.pooling_mode = pooling_mode
        self.projection = nn.Sequential(
            nn.LayerNorm(pose_channels),
            nn.Linear(pose_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_embed_dim),
        )

    def pool(self, pose_temporal_bundles: torch.Tensor) -> torch.Tensor:
        """[B, Nt, Lt, Cp, Hx, Wx] -> [B, Nt, Cp]. Separated from forward() so
        this can be swapped for attention pooling later without touching the
        projection MLP."""
        if self.pooling_mode != "mean":
            raise ValueError(
                f"Unsupported pooling_mode={self.pooling_mode!r}; only 'mean' "
                "is implemented at this stage."
            )
        return pose_temporal_bundles.mean(dim=(2, 4, 5))

    def forward(self, pose_temporal_bundles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        global_pose_raw = self.pool(pose_temporal_bundles)          # [B, Nt, Cp]
        global_pose_embedding = self.projection(global_pose_raw)    # [B, Nt, pose_embed_dim]
        return global_pose_raw, global_pose_embedding


class PoseReconstructionHead(nn.Module):
    """global_pose_embedding -> reconstructed pooled pose feature. Trains the
    GlobalPoseEncoder via a self-supervised reconstruction objective, without
    ever touching the visual branch."""

    def __init__(self, pose_embed_dim: int, hidden_dim: int, pose_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(pose_embed_dim),
            nn.Linear(pose_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_channels),
        )

    def forward(self, global_pose_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(global_pose_embedding)


# ---------------------------------------------------------------------------
# 3. Local Pose Occupancy
# ---------------------------------------------------------------------------

class LocalPoseOccupancy(nn.Module):
    """Generate one soft pose occupancy value for each spatio-temporal visual
    bundle. No learnable parameters -- purely deterministic given
    pose_aligned."""

    def __init__(
        self,
        temporal_patch_size: int,
        spatial_patch_height: int,
        spatial_patch_width: int,
        activation_mode: str = "abs_mean",
        normalization: str = "sigmoid",
        sigmoid_temperature: float = 0.01,
    ) -> None:
        super().__init__()
        if activation_mode not in ("abs_mean", "l2_norm", "max_abs"):
            raise ValueError(f"Unknown occupancy_activation_mode={activation_mode!r}")
        if normalization not in ("sigmoid", "minmax", "softmax_spatial", "none"):
            raise ValueError(f"Unknown occupancy_normalization={normalization!r}")
        self.lt = temporal_patch_size
        self.ph = spatial_patch_height
        self.pw = spatial_patch_width
        self.activation_mode = activation_mode
        self.normalization = normalization
        self.sigmoid_temperature = sigmoid_temperature

    def compute_activation(self, pose_aligned: torch.Tensor) -> torch.Tensor:
        """[B, T, Cp, Hx, Wx] -> [B, T, Hx, Wx]"""
        if self.activation_mode == "abs_mean":
            return pose_aligned.abs().mean(dim=2)
        if self.activation_mode == "l2_norm":
            return torch.sqrt(pose_aligned.pow(2).mean(dim=2) + 1e-8)
        if self.activation_mode == "max_abs":
            return pose_aligned.abs().amax(dim=2)
        raise AssertionError("unreachable")  # validated in __init__

    def patch_pool(self, pose_activation: torch.Tensor) -> torch.Tensor:
        """[B, T, Hx, Wx] -> [B, Nt, Nh, Nw, 1]"""
        b, t, h, w = pose_activation.shape
        lt, ph, pw = self.lt, self.ph, self.pw
        if t % lt != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(
                "LocalPoseOccupancy.patch_pool: T,H,W must be divisible by patch sizes. "
                f"got T={t}, H={h}, W={w}; temporal_patch_size={lt}, "
                f"spatial_patch_height={ph}, spatial_patch_width={pw}"
            )
        nt, nh, nw = t // lt, h // ph, w // pw
        x = pose_activation.contiguous()
        x = x.reshape(b, nt, lt, nh, ph, nw, pw)     # [B,Nt,Lt,Nh,Ph,Nw,Pw]
        x = x.permute(0, 1, 3, 5, 2, 4, 6)            # [B,Nt,Nh,Nw,Lt,Ph,Pw]
        x = x.contiguous()
        pooled = x.mean(dim=(4, 5, 6))                # [B,Nt,Nh,Nw]
        return pooled.unsqueeze(-1)                   # [B,Nt,Nh,Nw,1]

    def normalize(self, raw_occupancy: torch.Tensor) -> torch.Tensor:
        if self.normalization == "none":
            return raw_occupancy
        if self.normalization == "sigmoid":
            center = raw_occupancy.mean(dim=(2, 3), keepdim=True)
            return torch.sigmoid((raw_occupancy - center) / self.sigmoid_temperature)
        if self.normalization == "minmax":
            mn = raw_occupancy.amin(dim=(2, 3), keepdim=True)
            mx = raw_occupancy.amax(dim=(2, 3), keepdim=True)
            return (raw_occupancy - mn) / (mx - mn + 1e-8)
        if self.normalization == "softmax_spatial":
            b, nt, nh, nw, _ = raw_occupancy.shape
            flat = raw_occupancy.reshape(b, nt, nh * nw)
            sm = torch.softmax(flat, dim=-1)
            return sm.reshape(b, nt, nh, nw, 1)
        raise AssertionError("unreachable")  # validated in __init__

    def forward(self, pose_aligned: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        activation = self.compute_activation(pose_aligned)   # [B,T,Hx,Wx]
        raw_occupancy = self.patch_pool(activation)           # [B,Nt,Nh,Nw,1]
        occupancy = self.normalize(raw_occupancy)
        return activation, occupancy


# ---------------------------------------------------------------------------
# 4. Statistics / collapse diagnostics (pure functions, no state)
# ---------------------------------------------------------------------------

def _finite_or_warn(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        warnings.warn(f"[PoseBundleEncoder] {name} contains NaN/Inf")


def global_pose_embedding_stats(embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
    """embedding: [B, Nt, D]"""
    mean = embedding.mean()
    std = embedding.std()
    norm = embedding.norm(dim=-1).mean()
    channel_variance = embedding.var(dim=(0, 1), unbiased=False).mean()
    temporal_variance = (
        embedding.var(dim=1, unbiased=False).mean() if embedding.shape[1] > 1
        else torch.zeros((), device=embedding.device)
    )
    batch_variance = (
        embedding.var(dim=0, unbiased=False).mean() if embedding.shape[0] > 1
        else torch.zeros((), device=embedding.device)
    )

    if is_rank0():
        if std.item() < 1e-6:
            warnings.warn("[PoseBundleEncoder] global_pose_embedding std ~= 0 (collapse)")
        if embedding.shape[1] > 1 and temporal_variance.item() < 1e-6:
            warnings.warn("[PoseBundleEncoder] global_pose_temporal_variance ~= 0 "
                          "(embedding does not vary across temporal bundles)")
        if norm.item() > 100.0:
            warnings.warn(f"[PoseBundleEncoder] global_pose_norm is large ({norm.item():.2f})")
    _finite_or_warn("global_pose_embedding", embedding)

    return {
        "global_pose_mean": mean,
        "global_pose_std": std,
        "global_pose_norm": norm,
        "global_pose_channel_variance": channel_variance,
        "global_pose_temporal_variance": temporal_variance,
        "global_pose_batch_variance": batch_variance,
    }


def local_occupancy_stats(
    occupancy: torch.Tensor,
    active_threshold: float = 0.6,
    background_threshold: float = 0.4,
) -> Dict[str, torch.Tensor]:
    """occupancy: [B, Nt, Nh, Nw, 1]"""
    mean = occupancy.mean()
    std = occupancy.std()
    omin = occupancy.min()
    omax = occupancy.max()
    active_ratio = (occupancy > active_threshold).float().mean()
    background_ratio = (occupancy < background_threshold).float().mean()
    temporal_variance = (
        occupancy.var(dim=1, unbiased=False).mean() if occupancy.shape[1] > 1
        else torch.zeros((), device=occupancy.device)
    )
    spatial_variance = occupancy.var(dim=(2, 3), unbiased=False).mean()

    if is_rank0():
        if omax.item() < 1e-6:
            warnings.warn("[PoseBundleEncoder] local_pose_occupancy is ~all 0")
        if omin.item() > 1 - 1e-6:
            warnings.warn("[PoseBundleEncoder] local_pose_occupancy is ~all 1")
        if std.item() < 1e-6:
            warnings.warn("[PoseBundleEncoder] local_pose_occupancy std ~= 0 (spatially uniform)")
        if active_ratio.item() > 0.95:
            warnings.warn(f"[PoseBundleEncoder] occupancy_active_ratio is very high ({active_ratio.item():.3f})")
        if background_ratio.item() < 1e-6:
            warnings.warn("[PoseBundleEncoder] occupancy_background_ratio ~= 0")
        if occupancy.shape[1] > 1 and temporal_variance.item() < 1e-6:
            warnings.warn("[PoseBundleEncoder] occupancy_temporal_variance ~= 0")
    _finite_or_warn("local_pose_occupancy", occupancy)

    return {
        "occupancy_mean": mean,
        "occupancy_std": std,
        "occupancy_min": omin,
        "occupancy_max": omax,
        "occupancy_active_ratio": active_ratio,
        "occupancy_background_ratio": background_ratio,
        "occupancy_temporal_variance": temporal_variance,
        "occupancy_spatial_variance": spatial_variance,
    }


# ---------------------------------------------------------------------------
# 5. Top-level orchestrator
# ---------------------------------------------------------------------------

class PoseBundleEncoder(nn.Module):
    """Aligns pose features and produces global pose embeddings and local
    pose occupancy maps. Auxiliary branch only: never modulates visual_tokens
    or the UNet feature. See module docstring."""

    def __init__(
        self,
        pose_channels: int,
        pose_embed_dim: int,
        hidden_dim: int,
        temporal_patch_size: int,
        spatial_patch_height: int,
        spatial_patch_width: int,
        pose_normalization: str = "none",
        occupancy_activation_mode: str = "abs_mean",
        occupancy_normalization: str = "sigmoid",
        occupancy_sigmoid_temperature: float = 0.01,
        occupancy_active_threshold: float = 0.6,
        occupancy_background_threshold: float = 0.4,
    ) -> None:
        super().__init__()
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_height = spatial_patch_height
        self.spatial_patch_width = spatial_patch_width
        self.occupancy_active_threshold = occupancy_active_threshold
        self.occupancy_background_threshold = occupancy_background_threshold

        self.aligner = PoseFeatureAligner(pose_channels, normalization=pose_normalization)
        self.global_encoder = GlobalPoseEncoder(pose_channels, pose_embed_dim, hidden_dim)
        self.reconstruction_head = PoseReconstructionHead(pose_embed_dim, hidden_dim, pose_channels)
        self.local_occupancy = LocalPoseOccupancy(
            temporal_patch_size, spatial_patch_height, spatial_patch_width,
            activation_mode=occupancy_activation_mode,
            normalization=occupancy_normalization,
            sigmoid_temperature=occupancy_sigmoid_temperature,
        )

        if is_rank0():
            enc_params = sum(p.numel() for p in self.global_encoder.parameters())
            head_params = sum(p.numel() for p in self.reconstruction_head.parameters())
            aligner_params = sum(p.numel() for p in self.aligner.parameters())
            print(
                "[PoseBundleEncoder] "
                f"global_pose_encoder_params={enc_params:,}, "
                f"reconstruction_head_params={head_params:,}, "
                f"local_occupancy_params=0 (deterministic), "
                f"aligner_params={aligner_params:,} (0 unless channel_layernorm), "
                f"total={enc_params + head_params + aligner_params:,}"
            )

    def _to_temporal_bundles(self, pose_aligned: torch.Tensor) -> torch.Tensor:
        """[B,T,Cp,Hx,Wx] -> [B,Nt,Lt,Cp,Hx,Wx]"""
        b, t, c, h, w = pose_aligned.shape
        lt = self.temporal_patch_size
        if t % lt != 0:
            raise ValueError(
                f"PoseBundleEncoder: T={t} is not divisible by temporal_patch_size={lt}"
            )
        nt = t // lt
        return pose_aligned.reshape(b, nt, lt, c, h, w)

    def forward(
        self,
        pose_feature: torch.Tensor,
        target_spatial_size: Tuple[int, int],
        batch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if batch_size is None or num_frames is None:
            if pose_feature.dim() != 5:
                raise ValueError(
                    "batch_size/num_frames must be given explicitly when pose_feature "
                    f"is not already 5D (got dim={pose_feature.dim()})"
                )
            batch_size, num_frames = pose_feature.shape[0], pose_feature.shape[1]

        pose_aligned = self.aligner(pose_feature, batch_size, num_frames, target_spatial_size)
        pose_temporal_bundles = self._to_temporal_bundles(pose_aligned)

        nh = target_spatial_size[0] // self.spatial_patch_height
        nw = target_spatial_size[1] // self.spatial_patch_width
        nt = num_frames // self.temporal_patch_size

        if visual_tokens is not None:
            if visual_tokens.shape[0] != batch_size:
                raise ValueError(
                    f"PoseBundleEncoder: visual_tokens batch size {visual_tokens.shape[0]} "
                    f"!= pose batch size {batch_size}"
                )
            if visual_tokens.shape[1] != nt:
                raise ValueError(
                    f"PoseBundleEncoder: visual_tokens Nt={visual_tokens.shape[1]} != "
                    f"pose bundle Nt={nt} (num_frames={num_frames}, temporal_patch_size="
                    f"{self.temporal_patch_size} for pose; check visual/pose share the same "
                    "temporal_patch_size and the same underlying T)"
                )
            if tuple(visual_tokens.shape[2:4]) != (nh, nw):
                raise ValueError(
                    f"PoseBundleEncoder: visual_tokens (Nh,Nw)={tuple(visual_tokens.shape[2:4])} "
                    f"!= pose bundle (Nh,Nw)=({nh},{nw})"
                )

        global_pose_raw, global_pose_embedding = self.global_encoder(pose_temporal_bundles)
        pose_activation, local_pose_occupancy = self.local_occupancy(pose_aligned)

        reconstructed_global_pose = self.reconstruction_head(global_pose_embedding)
        pose_reconstruction_loss = F.smooth_l1_loss(
            reconstructed_global_pose.float(), global_pose_raw.detach().float()
        )

        global_stats = global_pose_embedding_stats(global_pose_embedding)
        occ_stats = local_occupancy_stats(
            local_pose_occupancy,
            active_threshold=self.occupancy_active_threshold,
            background_threshold=self.occupancy_background_threshold,
        )

        out = {
            "pose_aligned": pose_aligned,
            "pose_temporal_bundles": pose_temporal_bundles,
            "global_pose_raw": global_pose_raw,
            "global_pose_embedding": global_pose_embedding,
            "pose_activation": pose_activation,
            "local_pose_occupancy": local_pose_occupancy,
            "reconstructed_global_pose": reconstructed_global_pose,
            "pose_reconstruction_loss": pose_reconstruction_loss,
        }
        out.update(global_stats)
        out.update(occ_stats)
        return out


# ---------------------------------------------------------------------------
# 6. Pose shuffle diagnostic (not used in training loss -- diagnostic only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def pose_shuffle_diagnostic(
    encoder: PoseBundleEncoder,
    pose_feature: torch.Tensor,
    target_spatial_size: Tuple[int, int],
    batch_size: int,
    num_frames: int,
) -> Dict[str, float]:
    """Shuffles the temporal frame order of pose_feature and compares
    global_pose_embedding / local_pose_occupancy against the original.
    If the pose sequence changes but the outputs barely move, the pose
    branch is not actually responding to pose content."""
    pose_5d = encoder.aligner._restore_5d(pose_feature, batch_size, num_frames) \
        if pose_feature.dim() == 4 else pose_feature
    perm = torch.randperm(num_frames, device=pose_5d.device)
    pose_shuffled_5d = pose_5d[:, perm]

    out_orig = encoder(pose_5d, target_spatial_size, batch_size, num_frames)
    out_shuf = encoder(pose_shuffled_5d, target_spatial_size, batch_size, num_frames)

    emb_o, emb_s = out_orig["global_pose_embedding"], out_shuf["global_pose_embedding"]
    occ_o, occ_s = out_orig["local_pose_occupancy"], out_shuf["local_pose_occupancy"]

    embedding_mad = (emb_o - emb_s).abs().mean().item()
    cos_sim = F.cosine_similarity(emb_o.flatten(1), emb_s.flatten(1), dim=-1).mean().item()
    occupancy_mad = (occ_o - occ_s).abs().mean().item()

    if is_rank0() and embedding_mad < 1e-4 and occupancy_mad < 1e-4:
        warnings.warn(
            "[PoseBundleEncoder] pose_shuffle_diagnostic: shuffling frame order barely "
            f"changed the output (embedding_mad={embedding_mad:.2e}, "
            f"occupancy_mad={occupancy_mad:.2e}). The pose branch may not be responding "
            "to actual pose content."
        )

    return {
        "global_pose_embedding_mean_abs_diff": embedding_mad,
        "global_pose_cosine_similarity": cos_sim,
        "local_occupancy_mean_abs_diff": occupancy_mad,
    }
