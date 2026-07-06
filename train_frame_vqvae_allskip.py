"""
train_frame_vqvae_allskip.py

train_frame_vqvae.py 기반으로, 단일 skip이 아닌
4개 down_block 전체의 resnet 출력(총 8개 skip)에 VQ를 적용.

  블록별 VQ 모듈 (별도 codebook):
    vq[0] : down_blocks[0].resnets[0,1]  320ch  72x128
    vq[1] : down_blocks[1].resnets[0,1]  640ch  36x64
    vq[2] : down_blocks[2].resnets[0,1] 1280ch  18x32
    vq[3] : down_blocks[3].resnets[0,1] 1280ch   9x16

  Pass 1 (no_grad): UNet forward → 8개 skip 캡처 + 원본 출력 저장
  VQ (grad)       : 블록별로 2개 resnet 배치 concat → VQ → 복원
  Pass 2 (grad)   : 8개 skip 교체 → UNet 재실행 → e2e loss

실행:
    python train_frame_vqvae_allskip.py --config frame_vqvae_allskip_config.yaml
"""


from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from frame_vqvae_model_grouped import FrameVQVAE

LOGGER = logging.getLogger("train_frame_vqvae_allskip")


class CaptureHook:
    def __init__(self, module: nn.Module):
        self.output = None
        self._h = module.register_forward_hook(self._fn)

    def _fn(self, _m, _i, out):
        self.output = out[0] if isinstance(out, tuple) else out

    def remove(self):
        self._h.remove()


class ReplaceHook:
    def __init__(self, module: nn.Module):
        self.new_output = None
        self._h = module.register_forward_hook(self._fn)

    def _fn(self, _m, _i, out):
        if self.new_output is None:
            return out
        if isinstance(out, tuple):
            return (self.new_output,) + tuple(out[1:])
        return self.new_output

    def remove(self):
        self._h.remove()


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

    overfit_single = bool(cfg.get("vq_overfit_single_sample", False))
    fixed_batch = None
    if overfit_single:
        fixed_batch = next(iter(train_loader))
        LOGGER.info("Overfit mode: ON")

    overfit_fixed_sigma = bool(cfg.get("vq_overfit_fixed_sigma", False))
    if overfit_fixed_sigma:
        LOGGER.info("Fixed-sigma mode: ON")

    init_ckpt = str(cfg.init_checkpoint) if cfg.get("init_checkpoint") else None
    comps = load_components(str(cfg.base_model_path), device, init_checkpoint=init_ckpt)
    for m in [comps.vae, comps.image_encoder, comps.unet, comps.pose_net]:
        m.requires_grad_(False)
        m.eval()
    unet_dtype = next(comps.unet.parameters()).dtype

    n_off = 0
    for m in comps.unet.modules():
        if hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing:
            m.gradient_checkpointing = False
            n_off += 1
    LOGGER.info("Disabled gradient checkpointing on %d modules", n_off)

    # ── 4개 블록 타겟 모듈 설정 ─────────────────────────────────
    block_channels = list(comps.unet.config.block_out_channels)  # [320, 640, 1280, 1280]
    num_blocks = len(comps.unet.down_blocks)  # 4

    # 블록별 (resnet0, resnet1) 쌍
    target_pairs = []
    for bi in range(num_blocks):
        r0 = comps.unet.down_blocks[bi].resnets[0]
        r1 = comps.unet.down_blocks[bi].resnets[1]
        target_pairs.append((r0, r1))
        LOGGER.info("Block %d: ch=%d  resnets[0,1]", bi, block_channels[bi])

    # ── 블록별 VQ 모듈 (각각 별도 codebook) ─────────────────────
    e_dim      = int(cfg.get("vq_latent_dim", 64))
    n_e        = int(cfg.get("vq_num_codes",  512))
    beta       = float(cfg.get("vq_beta",     0.25))
    group_size = cfg.get("vq_group_size", None)

    vq_models = nn.ModuleList([
        FrameVQVAE(
            unet             = comps.unet,
            feature_channels = block_channels[bi],
            e_dim            = e_dim,
            n_e              = n_e,
            beta             = beta,
            group_size       = group_size,
        )
        for bi in range(num_blocks)
    ]).to(device)

    total_params = sum(p.numel() for p in vq_models.parameters())
    LOGGER.info("AllSkip VQ params: %d (%.2f M)", total_params, total_params / 1e6)

    optimizer = AdamW(vq_models.parameters(),
                      lr=float(cfg.get("vq_learning_rate", 1e-4)),
                      betas=(0.9, 0.999), weight_decay=1e-2)
    warmup = int(cfg.get("vq_lr_warmup_steps", 500))
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / warmup))

    noise_aug_strength = float(cfg.noise_aug_strength)
    vq_p_mean = float(cfg.get("vq_p_mean", 0.0))
    vq_p_std  = float(cfg.get("vq_p_std",  0.7))
    sigma_sample_mode = str(cfg.get("vq_sigma_sample_mode", "lognormal"))
    sigma_min = float(cfg.get("vq_sigma_min", 0.002))
    sigma_max = float(cfg.get("vq_sigma_max", 700.0))
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    if sigma_sample_mode == "loguniform":
        LOGGER.info("sigma sampling: loguniform [%.4f, %.1f]", sigma_min, sigma_max)

    w_skip  = float(cfg.get("vq_skip_weight", 0.01))
    num_codes = n_e

    global_step = 0
    fixed_noise_cache: dict = {}

    try:
        _num_epochs = int(cfg.vq_num_epochs) if "vq_num_epochs" in cfg else int(cfg.num_epochs)
        for _ in range(_num_epochs):
            vq_models.train()
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
                        return (rnd * vq_p_std + vq_p_mean).exp()

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
                    pose_latents = encode_pose_latents(comps, pose_images)

                    c_in    = 1.0 / (sigma ** 2 + 1.0) ** 0.5
                    c_noise = sigma.log() / 4.0
                    noisy_latents = video_latents + noise * sigma
                    input_latents = c_in * noisy_latents
                    latent_input  = torch.cat([input_latents, ref_image_latents], dim=2).to(unet_dtype)
                    added_time_ids = get_added_time_ids(
                        int(cfg.fps) - 1, int(cfg.motion_bucket_id),
                        noise_aug_strength, b, device, unet_dtype,
                    )
                    unet_kwargs = dict(
                        encoder_hidden_states=ref_embeds.to(unet_dtype),
                        added_time_ids=added_time_ids,
                        pose_latents=pose_latents.to(unet_dtype),
                        image_only_indicator=False,
                    )
                    c_noise_unet = c_noise.reshape(b).to(unet_dtype)
                    c_noise_1d   = c_noise.reshape(b).float()

                    # ── Pass 1: 8개 skip 캡처 + 원본 출력 저장 ──
                    cap_hooks = [(CaptureHook(r0), CaptureHook(r1))
                                 for r0, r1 in target_pairs]
                    out_orig = comps.unet(latent_input, c_noise_unet, **unet_kwargs).sample
                    for ch0, ch1 in cap_hooks:
                        ch0.remove()
                        ch1.remove()

                    # 블록별 raw feature 쌍 [B*2, f, ch, H, W] → model에 넘길 준비
                    raw_feat_pairs = [
                        (cap_hooks[bi][0].output.float().detach(),
                         cap_hooks[bi][1].output.float().detach())
                        for bi in range(num_blocks)
                    ]
                    out_orig = out_orig.float().detach()

                # ── VQ: 블록별로 2개 resnet concat → 복원 ──
                skip_recon_pairs = []
                vq_losses, recon_skip_losses, ppl_list, cb_list = [], [], [], []

                for bi, (vq_model, (f0, f1)) in enumerate(zip(vq_models, raw_feat_pairs)):
                    # 2개 resnet을 batch 방향으로 concat (같은 codebook)
                    feat_cat = torch.cat([f0, f1], dim=0)          # [2*B*f, ch, H, W]
                    recon_cat, vq_loss, ppl, indices, stats = vq_model(
                        feat_cat, b * 2, f, c_noise_1d.repeat(2)
                    )
                    recon0, recon1 = recon_cat.chunk(2, dim=0)

                    skip_recon_pairs.append((recon0.to(unet_dtype),
                                             recon1.to(unet_dtype)))

                    loss_rs = F.mse_loss(recon_cat.float(), feat_cat)
                    vq_losses.append(vq_loss)
                    recon_skip_losses.append(loss_rs)
                    ppl_list.append(stats["perplexity"])
                    cb_list.append(indices.unique().numel() / num_codes)

                # ── Pass 2: 8개 skip 교체 → UNet 재실행 ──
                rep_hooks = []
                for bi, (r0, r1) in enumerate(target_pairs):
                    rh0 = ReplaceHook(r0)
                    rh1 = ReplaceHook(r1)
                    rh0.new_output = skip_recon_pairs[bi][0]
                    rh1.new_output = skip_recon_pairs[bi][1]
                    rep_hooks.extend([rh0, rh1])

                out_recon = comps.unet(latent_input, c_noise_unet, **unet_kwargs).sample
                for rh in rep_hooks:
                    rh.remove()
                out_recon = out_recon.float()

                # ── Loss ──
                loss_vq_mean    = sum(vq_losses) / num_blocks
                loss_skip_mean  = sum(recon_skip_losses) / num_blocks
                loss_recon_out  = F.mse_loss(out_recon, out_orig)
                loss = w_skip * loss_skip_mean + loss_recon_out + loss_vq_mean

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vq_models.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                # ── 로깅 ──
                if global_step % int(cfg.print_every_steps) == 0:
                    avg_ppl = sum(ppl_list) / num_blocks
                    avg_cb  = sum(cb_list)  / num_blocks
                    per_block = "  ".join(
                        f"b{bi}:vq={vq_losses[bi].item():.4f}/ppl={ppl_list[bi]:.1f}/cb={cb_list[bi]*100:.1f}%"
                        for bi in range(num_blocks)
                    )
                    LOGGER.info(
                        "Step %d | total=%.5f  skip=%.5f  out=%.5f  vq=%.5f  ppl=%.1f  cb=%.1f%%",
                        global_step, loss.item(), loss_skip_mean.item(),
                        loss_recon_out.item(), loss_vq_mean.item(), avg_ppl, avg_cb * 100,
                    )
                    LOGGER.info("  %s", per_block)

                    writer.add_scalar("train/total",      loss.item(),            global_step)
                    writer.add_scalar("train/recon_skip", loss_skip_mean.item(),  global_step)
                    writer.add_scalar("train/recon_out",  loss_recon_out.item(),  global_step)
                    writer.add_scalar("train/vq_mean",    loss_vq_mean.item(),    global_step)
                    writer.add_scalar("train/ppl_mean",   avg_ppl,                global_step)
                    writer.add_scalar("train/cb_mean",    avg_cb,                 global_step)
                    for bi in range(num_blocks):
                        writer.add_scalar(f"train/b{bi}/vq",  vq_losses[bi].item(), global_step)
                        writer.add_scalar(f"train/b{bi}/ppl", ppl_list[bi],         global_step)
                        writer.add_scalar(f"train/b{bi}/cb",  cb_list[bi],          global_step)

                if global_step > 0 and global_step % int(cfg.save_every_steps) == 0:
                    sp = ckpt_dir / f"step_{global_step:08d}.pth"
                    torch.save({f"vq{bi}": vq_models[bi].state_dict()
                                for bi in range(num_blocks)}, sp)
                    LOGGER.info("Saved: %s", sp)

                global_step += 1

    finally:
        fp = ckpt_dir / f"last_{global_step:08d}.pth"
        torch.save({f"vq{bi}": vq_models[bi].state_dict()
                    for bi in range(num_blocks)}, fp)
        LOGGER.info("Final: %s", fp)
        writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/frame_vqvae_allskip_config.yaml")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    train(cfg)
