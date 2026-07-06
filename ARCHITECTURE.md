# Architecture — Continuous Pose-Conditioned Adapter for MimicMotion

**Branch**: `add_tokenizer`
**Last updated**: 2026-07-05

This document describes the auxiliary adapter pipeline built on top of the MimicMotion
(SVD UNet + PoseNet) baseline in this repo. Four stages, each independently toggled by a
config flag, each designed so that disabling it reproduces the previous stage's behavior
exactly (verified empirically, not just by inspection — see `test_adapter_baseline_invariance.py`).

```
enable_continuous_tokenizer          -> visual_tokenizer.py
enable_pose_bundle_encoder           -> pose_bundle_encoder.py
enable_pose_conditioned_film         -> pose_conditioned_film.py     (requires the above two)
enable_continuous_adapter_injection  -> continuous_adapter_injector.py (requires all three above)
```

## 1. Baseline (unchanged)

- **UNet**: `mimicmotion/modules/unet.py::UNetSpatioTemporalConditionModel` — SVD UNet,
  4 down_blocks (`block_out_channels=[320,640,1280,1280]`), trained with EDM/v-prediction
  preconditioning (`train_vq.py`'s `c_skip/c_out/c_in/loss_weight`, matching
  `EulerDiscreteScheduler`'s actual v-prediction formula — note `c_out` is negative).
- **PoseNet**: `mimicmotion/modules/pose_net.py::PoseNet` — `conv_layers` (→128ch) then
  `final_proj` (1x1 conv, 128→320ch) whose output is added into the UNet at `conv_in`.
- Both are still trained directly by the diffusion loss exactly as before; nothing in
  this document changes that path when all four flags above are `false`.

## 2. Stage 1 — Visual Tokenizer (`visual_tokenizer.py`)

Converts a selected UNet feature into non-overlapping spatio-temporal patch tokens and
back, as a **continuous** (non-VQ) bottleneck.

- **Selected feature**: `down_blocks[1]` output (`sample`, not `res_samples`), real shape
  `(B*T, 640, 32, 18)` for `resolution=576, num_frames=8` (confirmed via hook, not guessed).
- **`UNetFeatureCapture`**: generic persistent forward hook (reused for both this feature
  and `PoseNet.conv_layers`), stores `.output` reference every call, never modifies output.
- **`SpatioTemporalContinuousTokenizer`**:
  - `patchify`: `[B,T,C,H,W] → [B,Nt,Nh,Nw, Lt·Ph·Pw·C]` (`Nt=T/Lt, Nh=H/Ph, Nw=W/Pw`)
  - `encode`: `Linear(patch_dim → embed_dim)` (LayerNorm + Linear) → `visual_tokens`
  - `decode`: `Linear(embed_dim → patch_dim)`
  - `unpatchify`: inverse of `patchify` (exact permute/reshape, round-trip verified to
    `atol=1e-6`)
  - Default patch grid: `temporal_patch_size=4, spatial_patch_height=4, spatial_patch_width=3`
    → `patch_dim=30720`, `embed_dim=64` (≈480x compression bottleneck)
- **Loss**: `smooth_l1_loss(reconstructed, original_feature)`, weight `lambda_tokenizer_reconstruction`
  (`0.1` as of the adapter-injection stage; was `1.0` before diffusion_loss could be
  affected by the adapter).
- Never injected into the UNet at this stage in isolation — see Stage 4 for when it is.

## 3. Stage 2 — Pose Bundle Encoder (`pose_bundle_encoder.py`)

Aligns the pose branch's own feature to the visual feature's spatio-temporal grid and
produces two pose signals per bundle, entirely parallel to the visual branch.

- **Selected feature**: `PoseNet.conv_layers` output (before `final_proj`), real shape
  `(B*T, 128, 128, 72)`.
- **`PoseFeatureAligner`**: bilinear-interpolates to the visual feature's `(H,W)`, optional
  normalization (`none` / `channel_layernorm` / `channel_standardize`; `none` is the
  validated default — occupancy needs raw activation magnitude).
- **`GlobalPoseEncoder`**: mean-pools each temporal bundle → `LayerNorm→Linear→SiLU→Linear`
  → `global_pose_embedding [B,Nt,pose_embed_dim]` (default `pose_embed_dim=32`).
- **`PoseReconstructionHead`**: `global_pose_embedding → reconstructed pooled pose
  feature`, trains `GlobalPoseEncoder` via a self-supervised reconstruction loss
  (target detached) — this is the *only* signal training the global encoder pre-FiLM.
- **`LocalPoseOccupancy`**: deterministic, no learnable params. `abs_mean` activation over
  channels → patch-pool → `sigmoid((x - spatial_mean) / temperature)`.
  `occupancy_sigmoid_temperature=0.01` (**not 1.0** — real measurement showed `1.0`
  collapses occupancy to `occ_std≈0.002, active_ratio≈0`; `0.01` gives `occ_std≈0.18,
  active_ratio≈0.26`. See `CODE_REVIEW.md` §2.1 for a latent inconsistency in one of the
  code fallback defaults for this value.)
- **Loss**: `pose_reconstruction_loss` (self-supervised), weight `lambda_pose_reconstruction` (`0.1`).
- Collapse diagnostics (`global_pose_embedding_stats`, `local_occupancy_stats`) warn on
  std≈0, occupancy≈all-0/all-1, zero temporal variance, etc.

## 4. Stage 3 — Pose-Conditioned FiLM (`pose_conditioned_film.py`)

First stage where pose information actually modulates visual tokens (still never touches
the UNet in isolation).

```
modulated_tokens = visual_tokens
    + film_scale * local_pose_occupancy * (gamma * visual_tokens + beta)
gamma, beta = film_mlp(global_pose_embedding).chunk(2, dim=-1)
```

- `film_mlp`: `LayerNorm → Linear → SiLU → Linear(→ 2·visual_embed_dim)`, last Linear
  zero-initialized by default.
- `film_scale`: learnable scalar, `0.0` init by default → `modulated_tokens == visual_tokens`
  exactly at step 0 (`atol=1e-6`, verified).
- **Known, documented gradient-starvation trap**: `film_scale_init=0.0` **and**
  `film_last_linear_init="zero"` together (both defaults) make the gradient into
  `film_scale` and `film_mlp` **exactly zero**, not just small — reproduced on real
  project data. Fix used in this session: `film_scale_init: 1.0e-3` (keeps the init
  identity but unblocks gradient) — see the module's own docstring for the full
  derivation.
- Loss: `film_reg_loss = gamma.pow(2).mean() + beta.pow(2).mean()`, weight `lambda_film_reg` (`1e-4`).
- `pose_film_shuffle_diagnostic`: shuffles the temporal-bundle order of the pose signal
  and checks `modulated_tokens`/reconstruction actually change (logging only, never in
  the loss). Uses a cyclic shift rather than `randperm` — with `Nt` as small as 2 (this
  project's real config), a random permutation has a 50% chance of landing on the
  identity, which was observed in a real run and fixed.

## 5. Stage 4 — Continuous Adapter Residual Injection (`continuous_adapter_injector.py`)

First stage where `reconstructed_feature` is fed back into the UNet's actual forward
pass, and thus the first stage where `diffusion_loss` itself can be affected by the
adapter.

```
feature_delta      = reconstructed_feature - visual_feature
feature_for_unet    = visual_feature + injection_scale * feature_delta   # mode="residual"
                    = visual_feature                                     # mode="none"
```

- `injection_scale`: learnable scalar, `0.0` init → exact baseline identity at step 0.
- **Mechanism — a forward hook, not a UNet edit.** `down_blocks[1]` returns
  `(sample, res_samples)`; because this block has a downsampler, `res_samples[-1] IS
  sample` (same tensor object — verified against the installed `diffusers==0.37.0`
  source and the actual loaded model config, not assumed). A forward hook on
  `down_blocks[1]` can therefore replace *both* the continuing main path and its aliased
  skip-connection entry consistently, by returning a new `(feature_for_unet, res_samples[:-1]
  + (feature_for_unet,))` tuple — PyTorch substitutes a hook's non-`None` return value for
  the module's real output. If this aliasing assumption ever breaks (e.g. a different base
  UNet), the injector raises `RuntimeError` instead of silently misbehaving.
- **The entire tokenizer→pose→FiLM pipeline runs inside this one hook**, because
  injection has to happen *during* the same `unet(...)` call that produces `model_pred`
  — it cannot be computed afterward in the training loop the way the earlier stages were.
- **Auto batch/frame detection**: a separate forward *pre*-hook on the top-level `unet`
  reads `(batch_size, num_frames)` directly from the incoming 5D `sample` tensor on every
  call. This matters because `mimicmotion/pipelines/pipeline_mimicmotion.py`'s real
  inference loop calls the UNet very differently from training: once per CFG branch
  (`batch_size=1` each) per tile, with `num_frames` varying per tile — the pre-hook makes
  this transparent without the caller needing to track it.
- Loss: `injection_reg_loss = injection_scale.pow(2)`, weight `lambda_injection_reg` (`1e-4`),
  only added to `total_loss` when `adapter_injection_mode="residual"`.
- **Known open issue** (see `CODE_REVIEW.md` §2.2): the CFG-unconditional branch at
  inference also receives injection (same captured pose feature, since both UNet calls
  per tile share it), which was not addressed by any spec in this session and should be
  decided on before drawing strong conclusions from inference-quality comparisons.

## 6. Data flow (training loop, `enable_continuous_adapter_injection=true`)

```
pose_images ──> encode_pose_latents (PoseNet) ──> pose_latents ──┐  (captures pose_feature_capture.output)
                                                                    │
video_latents, ref_embeds, sigma, noise ──> latent_input ─────────┤
                                                                    ▼
                                                          comps.unet(latent_input, ..., pose_latents=pose_latents)
                                                                    │
                                                     down_blocks[0] │
                                                                    ▼
                                              ┌── down_blocks[1] ──┴─────────────────────────┐
                                              │   sample, res_samples = block(...)            │
                                              │   [ContinuousAdapterInjector hook fires here] │
                                              │     visual_feature = restore_5d(sample)        │
                                              │     visual_tokens  = tokenizer.encode(patchify(visual_feature))
                                              │     pose_output    = pose_bundle_encoder(pose_feature_capture.output, ...)
                                              │     film_output    = pose_conditioned_film(visual_tokens, pose_output[...])
                                              │     reconstructed  = tokenizer.unpatchify(tokenizer.decode(film_output["modulated_tokens"]))
                                              │     feature_for_unet = visual_feature + injection_scale * (reconstructed - visual_feature)
                                              │     return (feature_for_unet_4d, res_samples[:-1] + (feature_for_unet_4d,))
                                              └────────────────────────────────────────────────┘
                                                                    │
                                                     down_blocks[2..3], mid_block, up_blocks, conv_out
                                                                    ▼
                                                              model_pred
                                                                    │
                              v-prediction preconditioning ────────┤
                                                                    ▼
                                                             diffusion_loss
                                                                    │
   total_loss = diffusion_loss
              + lambda_tokenizer_reconstruction * tokenizer_output["reconstruction_loss"]
              + lambda_pose_reconstruction      * pose_output["pose_reconstruction_loss"]
              + lambda_film_reg                 * film_reg_loss
              + lambda_injection_reg            * injection_reg_loss   (only if mode="residual")
```

When `enable_continuous_adapter_injection=false`, the same four pieces (tokenizer, pose
bundle encoder, FiLM) are instead computed manually in the training loop *after*
`model_pred` is produced, reading from plain (non-injecting) `UNetFeatureCapture` hooks —
numerically identical results, just relocated, since nothing is fed back into the UNet in
that mode.

## 7. Config flags (in `configs/train_config.yaml`)

| Flag | Requires | Effect when `true` |
|---|---|---|
| `enable_continuous_tokenizer` | — | patchify/encode/decode/unpatchify auxiliary branch |
| `enable_pose_bundle_encoder` | — | global pose embedding + local occupancy auxiliary branch |
| `enable_pose_conditioned_film` | tokenizer + pose_bundle | FiLM-modulates tokens before decode |
| `enable_continuous_adapter_injection` | tokenizer + pose_bundle + FiLM | residual-injects into UNet forward (see `adapter_injection_mode`) |
| `adapter_injection_mode` | injection enabled | `"none"` (compute only) or `"residual"` (actually inject) |

Each stage's own learning rate is a **required** (no fallback) config key, its own
optimizer param group: `tokenizer_learning_rate`, `pose_encoder_learning_rate`,
`film_learning_rate`, `injection_learning_rate`.

## 8. Checkpoint format

A single flat `state_dict` saved by `_trainable_state_dict()` in `train_vq.py`:

```
unet.*                          (always)
pose_net.*                      (always)
continuous_tokenizer.*          (if enable_continuous_tokenizer)
pose_bundle_encoder.*           (if enable_pose_bundle_encoder)
pose_conditioned_film.*         (if enable_pose_conditioned_film)
continuous_adapter_injector.*   (if enable_continuous_adapter_injection — just `injection_scale`)
```

The stock `mimicmotion/utils/loader.py::create_pipeline` (used by `inference.py` /
`inference_trained.py`) loads this with `strict=False`, so the extra prefixed keys are
silently ignored (logged as "Unexpected key") and inference works with the plain
baseline pipeline regardless of which stages were enabled at training time. To actually
*exercise* the adapter at inference (not just train it), a separate script,
`inference_with_adapter.py`, reconstructs the four modules from those extra keys and
re-attaches `ContinuousAdapterInjector` to the loaded pipeline's `unet`/`pose_net`.

## 9. File map

| File | Role |
|---|---|
| `visual_tokenizer.py` | Stage 1 module + `UNetFeatureCapture` + B↔BT reshape helpers |
| `pose_bundle_encoder.py` | Stage 2 module + collapse diagnostics + `pose_shuffle_diagnostic` (currently dead code, see `CODE_REVIEW.md` §2.3) |
| `pose_conditioned_film.py` | Stage 3 module + `pose_film_shuffle_diagnostic` |
| `continuous_adapter_injector.py` | Stage 4 module (the hook-based injector) |
| `train_vq.py` | training loop wiring all four stages + config validation cascade |
| `inference_with_adapter.py` | reconstructs + re-attaches the adapter for real inference |
| `test_visual_tokenizer.py`, `test_pose_bundle_encoder.py`, `test_pose_conditioned_film.py`, `test_continuous_adapter_injector.py` | standalone unit tests, no project deps beyond the module under test |
| `test_adapter_baseline_invariance.py` | real-model A/B/C baseline-invariance check |
| `CODE_REVIEW.md` | issue tracker / review notes for this pipeline |
| `ARCHITECTURE.md` | this file |

## 10. Verification log

**2026-07-05**: every claim above re-checked against the actual source (not re-derived
from this document) — `unet.py` L434/442-444/450-465/502, `train_vq.py`'s config-flag
wiring and the stale `occupancy_sigmoid_temperature=1.0` fallback at line 190 (still
present, see `CODE_REVIEW.md` §2.1), `configs/train_config.yaml`'s actual values, and the
`pipeline_mimicmotion.py` dual `self.unet(...)` CFG call. The four lightweight test
suites (`test_visual_tokenizer.py`, `test_pose_bundle_encoder.py`,
`test_pose_conditioned_film.py`, `test_continuous_adapter_injector.py`, 19 tests total)
were executed live (CPU-only, to avoid the GPUs busy with running jobs) and all passed;
`test_continuous_adapter_injector.py`'s printed real feature shape `(8, 640, 32, 18)`
matches this document's claim. `test_adapter_baseline_invariance.py` was not re-run here
(needs real UNet/VAE weights on GPU) — taken on the existing record. No discrepancies
found; no content changes were needed.
