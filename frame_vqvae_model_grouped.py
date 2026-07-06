"""
frame_vqvae_model_grouped.py

MimicMotion UNet 의 down_block skip feature 를 VQ 로 이산화/복원하는 모듈.
train_frame_vqvae_allskip.py 에서 4개 down_block (총 8개 resnet 출력) 에
블록별로 하나씩 붙여서 사용.

동작 (frame 직접 VQ, delta/cumsum 없음):
  skip feature X → std 정규화 → [X, t_emb] concat → encoder(1x1 conv) → z_e
                 → VectorQuantizer → z_q → decoder → X_hat → 역정규화
  복원된 skip 을 UNet 의 원래 skip 자리에 재주입 (학습 코드의 Pass 2)

구성 요소:
  - VectorQuantizer : https://github.com/MishaLaskin/vqvae (models/quantizer.py)
                      단, codebook/commitment 항의 beta 위치를 원 논문
                      (van den Oord 2017) 순서로 정정 (아래 주석 참고).
  - ResidualLayer   : 같은 저장소 models/residual.py 원본 그대로.
  - encoder/decoder : 최소 구성으로 자체 추가.

group_size 옵션:
  NF 프레임을 group_size 단위 시간 윈도우로 나눠 각 윈도우를 독립적으로 VQ.
  codebook 은 전체가 공유.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return x + self.res_block(x)


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, dtype=torch.float32, device=device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
            torch.mean((z_q.detach() - z) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class FrameVQVAE(nn.Module):

    def __init__(self, unet, feature_channels, e_dim=64, n_e=512, beta=0.25,
                 group_size=None):
        super().__init__()
        self.feature_channels = feature_channels
        self.e_dim = e_dim
        self.n_e = n_e
        self.group_size = group_size

        # frozen time embedding (noise-level context)
        self.time_proj      = copy.deepcopy(unet.time_proj)
        self.time_embedding = copy.deepcopy(unet.time_embedding)
        self.time_proj.requires_grad_(False)
        self.time_embedding.requires_grad_(False)
        time_embed_dim      = unet.time_embedding.linear_2.out_features
        self.time_emb_proj  = nn.Linear(time_embed_dim, feature_channels)
        self.t_emb_norm     = nn.LayerNorm(feature_channels)

        # encoder: [delta, t_emb] (2C) → e_dim
        self.encoder = nn.Conv2d(feature_channels * 2, e_dim, kernel_size=1)

        # 공식 VectorQuantizer (그룹이 달라도 동일한 codebook 공유)
        self.quantizer = VectorQuantizer(n_e, e_dim, beta)

        # decoder: z_q (e_dim) → delta 복원 (C)  [깊게: 공식 ResidualLayer 사용]
        h_dim = feature_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(e_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualLayer(h_dim, h_dim, h_dim // 2),
            ResidualLayer(h_dim, h_dim, h_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(h_dim, feature_channels, kernel_size=1),
        )

    def _quantize_grouped(self, z_e, B, NF):
        gs = self.group_size
        e_dim, H, W = z_e.shape[1], z_e.shape[-2], z_e.shape[-1]
        z_e = z_e.view(B, NF, e_dim, H, W)

        z_q_list, loss_list, ppl_list, idx_list = [], [], [], []
        for start in range(0, NF, gs):
            end = min(start + gs, NF)
            # 모든 샘플의 프레임 [start:end] 를 하나의 그룹으로 VQ
            z_e_g = z_e[:, start:end].reshape(-1, e_dim, H, W)
            loss_g, z_q_g, ppl_g, _, idx_g = self.quantizer(z_e_g)
            z_q_list.append(z_q_g.view(B, end - start, e_dim, H, W))
            loss_list.append(loss_g)
            ppl_list.append(ppl_g)
            idx_list.append(idx_g)

        z_q      = torch.cat(z_q_list, dim=1).reshape(B * NF, e_dim, H, W)
        vq_loss  = sum(loss_list) / len(loss_list)
        ppl      = sum(ppl_list) / len(ppl_list)
        indices  = torch.cat(idx_list, dim=0)
        return vq_loss, z_q, ppl, indices

    def forward(self, skip, B, NF, c_noise):
    
        C, H, W = self.feature_channels, skip.shape[-2], skip.shape[-1]

        X = skip.view(B, NF, C, H, W)                       # [B, NF, C, H, W]

        # frame 을 스칼라 std 로 정규화 (가역적)
        x_scale = X.detach().std().clamp(min=1e-5)
        X_n     = X / x_scale

        # time embedding → [B, C] → broadcast (NF 프레임 전부)
        t = self.time_proj(c_noise).to(dtype=X.dtype)
        t = self.time_embedding(t)
        t = self.time_emb_proj(t)
        t = self.t_emb_norm(t)
        t = t[:, None, :, None, None].expand(B, NF, C, H, W)

        # encoder 입력 z_e : [B*NF, e_dim, H, W]
        z_in = torch.cat([X_n, t], dim=2).view(B * NF, 2 * C, H, W)
        z_e  = self.encoder(z_in)

        # ── VectorQuantizer (그룹 분할 or 전체) ──
        if self.group_size is not None and self.group_size < NF:
            vq_loss, z_q, perplexity, indices = self._quantize_grouped(z_e, B, NF)
        else:
            vq_loss, z_q, perplexity, _, indices = self.quantizer(z_e)

        # ── decoder: z_q → frame 직접 복원 (cumsum 없음) ──
        X_hat_n = self.decoder(z_q).view(B, NF, C, H, W)
        X_recon = X_hat_n * x_scale                         # 정규화 역변환
        skip_recon = X_recon.reshape(B * NF, C, H, W)

        # ── 지표 ──
        with torch.no_grad():
            recon_frame = F.mse_loss(X_hat_n, X_n).item()
            stats = {
                "vq":          vq_loss.item(),
                "perplexity":  perplexity.item(),
                "recon_frame": recon_frame,
            }

        return skip_recon, vq_loss, perplexity, indices, stats