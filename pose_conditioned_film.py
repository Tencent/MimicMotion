"""
pose_conditioned_film.py

Global Pose Embedding으로부터 FiLM(gamma, beta)을 생성하고, Local Pose
Occupancy로 modulation 위치를 게이팅하여 visual token을 처음으로
pose-conditioned하게 변조하는 모듈.

이 단계에서는 modulated token을 다시 tokenizer.decode()/unpatchify()하여
feature reconstruction loss를 계산하는 것까지만 한다. 다음은 아직 구현하지
않는다: UNet feature에 reconstructed feature 주입, injection_scale, diffusion
loss 구조 변경, VQ, codebook, commitment loss, hard/soft quantization.

film_scale이 0으로 초기화되어 있으면(기본값) modulated_tokens는 학습 초기에
visual_tokens와 정확히 동일하다(atol=1e-6) -- 이 항등성이 깨지면 기존 visual
tokenizer/pose bundle encoder 단계의 baseline이 깨진다. 이는
test_pose_conditioned_film.py에서 명시적으로 검증한다.

주의(2026-07-04, 실제 계산으로 확인, 추측 아님): film_scale=0.0 AND 마지막
Linear zero-init을 동시에 사용하면(둘 다 기본값), modulated_tokens는 항상
visual_tokens와 정확히 같을 뿐 아니라, film_mlp/film_scale로 흘러가는
gradient도 정확히 0이 된다 -- film_delta = gamma*visual_tokens+beta가 이미
0이므로 d(modulated)/d(gamma_weight) = film_scale * occupancy * (...) = 0.
즉 backward()는 정상 실행되고 .grad는 None이 아니지만, 그 값 자체는 모두 0이다
(파라미터가 계산 그래프에는 연결되어 있으나 이 특정 초기값에서 국소
gradient가 0인 경우 -- 초기화가 잘못된 것이 아니라 이중 zero-init의 수학적
필연적 결과). 실제로 학습이 진행되려면(단순 backward 호출이 아니라) 다음 중
하나가 필요하다: (a) film_scale_init을 아주 작은 양수(예: 1e-3)로 설정 (마지막
Linear는 zero-init 유지 -- 이 경우 film_delta의 "값"은 여전히 0이지만
gamma_weight에 대한 "국소 미분"은 film_scale(!=0) 때문에 0이 아니게 되어 학습이
시작될 수 있다), 또는 (b) last_linear_init="small_normal"로 gamma/beta 자체를
0이 아니게 초기화. 기본값은 사용자 스펙대로 film_scale_init=0.0,
last_linear_init="zero"를 유지한다(가장 안전 -- baseline과 완전히 동일하게
시작). config로 위 대안을 선택할 수 있게 한다(train_vq.py의
film_scale_init/film_last_linear_init 참고).
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from visual_tokenizer import is_rank0


class PoseConditionedFiLM(nn.Module):
    """Generate gamma/beta from global_pose_embedding and apply
    local-occupancy-gated FiLM modulation to visual tokens.

        modulated_tokens = visual_tokens
            + film_scale * local_pose_occupancy * (gamma * visual_tokens + beta)

    Inputs:
        visual_tokens:         [B, Nt, Nh, Nw, Dv]
        global_pose_embedding: [B, Nt, Dp]
        local_pose_occupancy:  [B, Nt, Nh, Nw, 1]
    Outputs:
        modulated_tokens: [B, Nt, Nh, Nw, Dv]
        gamma, beta:      [B, Nt, Dv]
    """

    def __init__(
        self,
        visual_embed_dim: int,
        pose_embed_dim: int,
        hidden_dim: int,
        film_scale_init: float = 0.0,
        last_linear_init: str = "zero",
        last_linear_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if last_linear_init not in ("zero", "small_normal"):
            raise ValueError(
                f"Unknown last_linear_init={last_linear_init!r}; "
                "expected 'zero' or 'small_normal'"
            )
        self.visual_embed_dim = visual_embed_dim
        self.pose_embed_dim = pose_embed_dim

        self.film_mlp = nn.Sequential(
            nn.LayerNorm(pose_embed_dim),
            nn.Linear(pose_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * visual_embed_dim),
        )
        last_linear = self.film_mlp[-1]
        if last_linear_init == "zero":
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        else:
            nn.init.normal_(last_linear.weight, std=last_linear_init_std)
            nn.init.zeros_(last_linear.bias)

        self.film_scale = nn.Parameter(torch.tensor(float(film_scale_init)))

        if is_rank0():
            params = sum(p.numel() for p in self.parameters())
            print(
                "[PoseConditionedFiLM] "
                f"visual_embed_dim={visual_embed_dim}, pose_embed_dim={pose_embed_dim}, "
                f"hidden_dim={hidden_dim}, last_linear_init={last_linear_init}, "
                f"film_scale_init={film_scale_init}, params={params:,}"
            )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        global_pose_embedding: torch.Tensor,
        local_pose_occupancy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if visual_tokens.dim() != 5:
            raise ValueError(
                "PoseConditionedFiLM expects visual_tokens as 5D [B,Nt,Nh,Nw,Dv], "
                f"got shape={tuple(visual_tokens.shape)}"
            )
        if visual_tokens.shape[-1] != self.visual_embed_dim:
            raise ValueError(
                f"PoseConditionedFiLM: visual_tokens last dim {visual_tokens.shape[-1]} "
                f"!= visual_embed_dim={self.visual_embed_dim}"
            )
        if global_pose_embedding.dim() != 3 or global_pose_embedding.shape[-1] != self.pose_embed_dim:
            raise ValueError(
                "PoseConditionedFiLM expects global_pose_embedding as 3D [B,Nt,Dp] with "
                f"Dp={self.pose_embed_dim}, got shape={tuple(global_pose_embedding.shape)}"
            )
        if local_pose_occupancy.dim() != 5 or local_pose_occupancy.shape[-1] != 1:
            raise ValueError(
                "PoseConditionedFiLM expects local_pose_occupancy as 5D [B,Nt,Nh,Nw,1], "
                f"got shape={tuple(local_pose_occupancy.shape)}"
            )
        b, nt, nh, nw, _dv = visual_tokens.shape
        if tuple(global_pose_embedding.shape[:2]) != (b, nt):
            raise ValueError(
                "PoseConditionedFiLM: global_pose_embedding (B,Nt)="
                f"{tuple(global_pose_embedding.shape[:2])} != visual_tokens (B,Nt)=({b},{nt})"
            )
        if tuple(local_pose_occupancy.shape[:4]) != (b, nt, nh, nw):
            raise ValueError(
                "PoseConditionedFiLM: local_pose_occupancy (B,Nt,Nh,Nw)="
                f"{tuple(local_pose_occupancy.shape[:4])} != visual_tokens (B,Nt,Nh,Nw)="
                f"({b},{nt},{nh},{nw})"
            )

        film_params = self.film_mlp(global_pose_embedding)   # [B, Nt, 2*Dv]
        gamma, beta = film_params.chunk(2, dim=-1)            # each [B, Nt, Dv]

        gamma_b = gamma[:, :, None, None, :]                  # [B, Nt, 1, 1, Dv]
        beta_b = beta[:, :, None, None, :]

        film_delta = gamma_b * visual_tokens + beta_b         # [B, Nt, Nh, Nw, Dv]
        modulated_tokens = visual_tokens + self.film_scale * local_pose_occupancy * film_delta

        return {
            "modulated_tokens": modulated_tokens,
            "gamma": gamma,
            "beta": beta,
            "film_delta": film_delta,
        }


@torch.no_grad()
def pose_film_shuffle_diagnostic(
    film: PoseConditionedFiLM,
    visual_tokens: torch.Tensor,
    global_pose_embedding: torch.Tensor,
    local_pose_occupancy: torch.Tensor,
    decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """같은 visual_tokens에 대해 원래 pose와, temporal bundle(Nt) 축으로 섞은
    pose를 넣어 modulated_tokens(및 선택적으로 reconstruction)이 실제로
    달라지는지 확인한다. 학습 loss에는 쓰지 않고 logging 전용이다.

    decode_fn이 주어지면(예: `lambda t: tokenizer.unpatchify(tokenizer.decode(t), shape)`)
    reconstruction 공간에서의 차이도 함께 계산한다.

    2026-07-04, 실제 500-step debug overfit 실행으로 확인: 이 프로젝트의 실제
    config(num_frames=8, temporal_patch_size=4)에서는 Nt=2뿐이라
    torch.randperm(2)가 절반의 확률로 항등순열([0,1])을 반환해 진단이 매
    호출마다 무작위로 무의미해지는 문제가 있었다(로그에서
    shuffle_mod_l1=0.00000이 간헐적으로 관찰됨 -- 버그가 아니라 우연히 순서가
    안 섞인 것). torch.randperm 대신 cyclic shift(모든 원소가 반드시 이동,
    Nt>1이면 고정점 없음)를 사용해 매번 확실히 섞이도록 한다.
    """
    nt = global_pose_embedding.shape[1]
    if nt <= 1:
        warnings.warn(
            "[PoseConditionedFiLM] pose_film_shuffle_diagnostic: Nt<=1, shuffling is a no-op"
        )
    perm = torch.cat([torch.arange(1, nt), torch.arange(0, 1)]).to(global_pose_embedding.device)
    shuffled_embedding = global_pose_embedding[:, perm]
    shuffled_occupancy = local_pose_occupancy[:, perm]

    out_orig = film(visual_tokens, global_pose_embedding, local_pose_occupancy)
    out_shuf = film(visual_tokens, shuffled_embedding, shuffled_occupancy)

    modulated_l1_diff = (out_orig["modulated_tokens"] - out_shuf["modulated_tokens"]).abs().mean().item()
    result: Dict[str, float] = {"shuffle_modulated_token_l1_diff": modulated_l1_diff}

    if decode_fn is not None:
        recon_orig = decode_fn(out_orig["modulated_tokens"])
        recon_shuf = decode_fn(out_shuf["modulated_tokens"])
        result["shuffle_reconstruction_l1_diff"] = (recon_orig - recon_shuf).abs().mean().item()

    if is_rank0() and modulated_l1_diff < 1e-6:
        warnings.warn(
            "[PoseConditionedFiLM] pose_film_shuffle_diagnostic: shuffling pose temporal "
            f"order barely changed modulated_tokens (l1_diff={modulated_l1_diff:.2e}). "
            "The FiLM branch may not be responding to pose content (film_scale still ~0?, "
            "gamma/beta collapsed?)."
        )
    return result
