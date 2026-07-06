# Code Review — Continuous Adapter Pipeline (tokenizer → pose bundle → FiLM → residual injection)

**Date**: 2026-07-05
**Branch**: `add_tokenizer`
**Scope**: `visual_tokenizer.py`, `pose_bundle_encoder.py`, `pose_conditioned_film.py`,
`continuous_adapter_injector.py`, `train_vq.py`, `configs/train_config.yaml`,
`inference_with_adapter.py`, `configs/inference_adapter.yaml`, and their `test_*.py` files.

## 1. What this code does (for context)

Four auxiliary-branch stages, each gated by its own config flag, layered on top of the
EDM/v-prediction SVD training baseline:

1. **`visual_tokenizer.py`** — patchify a selected UNet feature (`down_blocks[1]` output)
   into spatio-temporal bundles, encode/decode through a continuous (non-VQ) bottleneck.
2. **`pose_bundle_encoder.py`** — align `PoseNet.conv_layers` output to the visual
   feature's grid, produce a Global Pose Embedding + Local Pose Occupancy per bundle.
3. **`pose_conditioned_film.py`** — FiLM-modulate visual tokens using the pose embedding
   (gated by occupancy), gated by a learnable `film_scale` starting at 0.
4. **`continuous_adapter_injector.py`** — residually inject the reconstructed feature
   back into the UNet's actual forward pass via a `down_blocks[1]` forward hook, gated
   by a learnable `injection_scale` starting at 0.

Each stage is designed so that disabling it (or its scale being 0) reproduces the
previous stage's behavior exactly — this was verified empirically at each step (see
`test_adapter_baseline_invariance.py` and the various `debug_overfit_*` runs logged in
this session).

## 2. Findings, ranked by severity

### 2.1 HIGH — `train_vq.py:190` uses a stale, dangerous fallback default for `occupancy_sigmoid_temperature`

```python
occupancy_sigmoid_temperature=float(cfg.get("occupancy_sigmoid_temperature", 1.0)),
```

Every other place in the codebase now defaults this to **0.01**:
`pose_bundle_encoder.py:371` (`PoseBundleEncoder.__init__`'s own default), `inference_with_adapter.py:84`,
`test_adapter_baseline_invariance.py:144`, and `configs/train_config.yaml:149` (which sets it
explicitly, with a long comment documenting *why* — real measurement showed `temperature=1.0`
collapses occupancy to `occ_std=0.002, active_ratio=0.00`, and `0.01` was the value chosen
after a real sweep). Only `train_vq.py` line 190 still has the old `1.0` fallback.

**Impact**: harmless today because `configs/train_config.yaml` always sets the key
explicitly. But it's a landmine for any future config that omits the key (e.g. a new
experiment config copied without every comment) — training would silently run with the
one temperature value already proven to collapse local pose occupancy, with no error or
warning to indicate why.

**Fix**: change the fallback in `train_vq.py:190` to `0.01` to match every other
definition of this default in the codebase.

### 2.2 MEDIUM — CFG-negative (unconditional) branch also receives pose-conditioned injection at inference time

`mimicmotion/pipelines/pipeline_mimicmotion.py`'s denoising loop calls `self.unet(...)`
twice per tile per timestep: once for the "classifier-free" branch with `pose_latents=None`,
and once for the "normal" branch with real `pose_latents`. Both calls pass through the
same `down_blocks[1]`, so **both** fire `ContinuousAdapterInjector`'s hook, and both use
the *same* captured `pose_feature_capture.output` (from the single upstream
`pose_net(image_pose[idx])` call that precedes both).

This means that once `adapter_injection_mode="residual"` and `injection_scale != 0`, the
"unconditional" CFG branch is no longer purely unconditional — it also receives a
pose-conditioned residual injection, even though its own `conv_in` never added
`pose_latents`. This narrows the conditional/unconditional contrast CFG guidance is
supposed to amplify, in a way none of this session's specs anticipated (they were written
against the training loop, which only ever calls the UNet once per step with real pose
latents).

**Impact**: not a crash, not a shape bug — a semantic side effect discovered by reading
`pipeline_mimicmotion.py`'s actual tiling/CFG code (not by guessing). It may partially
explain why the visual difference between baseline and continuous-injection generations
was subtle in the tested run (small `injection_scale`, but also a CFG contrast that's
being uniformly boosted on both branches rather than sharpened).

**Suggested next step**: decide deliberately whether the uncond branch should skip
injection entirely (e.g. gate on whether `pose_latents is None` was passed into the same
UNet call) before relying on this for any real generation-quality claims. Not fixed in
this session because it wasn't part of any explicit spec — flagging for a decision.

### 2.3 MEDIUM — `pose_shuffle_diagnostic` (pose_bundle_encoder.py) is imported but never called

```python
# train_vq.py:31
from pose_bundle_encoder import PoseBundleEncoder, pose_shuffle_diagnostic
```

Confirmed via `grep` — this name appears nowhere else in `train_vq.py`. It has its own
standalone test (`test_pose_bundle_encoder.py`) and works correctly in isolation, but was
never wired into the training loop's periodic logging the way its later counterpart
(`pose_film_shuffle_diagnostic`, wired in at `train_vq.py:788-807`) was. This looks like a
stage that was implemented and tested but the "wire it into real training logging" step
was skipped — worth either wiring it in (cheap: same pattern as the FiLM one) or removing
the dead import.

### 2.4 LOW — `ContinuousAdapterInjector.injection_mode` can be reassigned post-construction without validation

The constructor validates `injection_mode in ("none", "residual")`, but it's a plain
string attribute, not a validated property. `inference_with_adapter.py:171` does:

```python
adapter_injector.injection_mode = "residual"
```

directly, bypassing that check. Today this is safe (single hardcoded literal, no typo),
but if this pattern is copied elsewhere with a typo (e.g. `"residuals"`), the hook's
`if self.injection_mode == "residual": ... else: ...` silently falls through to the
"no injection" branch with **no error at all** — the exact kind of silent misconfiguration
the rest of this codebase has been careful to reject loudly (e.g.
`adapter_injection_mode` *is* validated at parse time in `train_vq.py`). Consider adding
a small `@injection_mode.setter`-style validation to `ContinuousAdapterInjector` for
defense in depth.

### 2.5 LOW — implicit dtype coupling in `inference_with_adapter.py`

`torch.set_default_dtype(torch.float16)` is called once at the top of `main()`, and the
adapter modules (`SpatioTemporalContinuousTokenizer`, `PoseBundleEncoder`,
`PoseConditionedFiLM`) are constructed *after* that point with no explicit `dtype=` — so
they end up fp16 only because of this ordering. If someone reorders the code (e.g. builds
the adapter modules before setting the default dtype, or refactors `main()`), they would
silently become fp32 and mismatch the fp16 pipeline's tensors, most likely producing a
runtime `RuntimeError` on the first matmul (fails loudly, not silently — so the risk is
"confusing error", not "wrong output") — still worth an explicit `dtype=torch.float16` at
construction rather than relying on ambient global state.

### 2.6 LOW — `feature_capture` (the read-only `UNetFeatureCapture` on `down_blocks[1]`) becomes vestigial once `enable_continuous_adapter_injection=True`

`train_vq.py:161-162` unconditionally creates `feature_capture` whenever the tokenizer or
pose branch is enabled. When adapter injection is *also* enabled, nothing in the training
loop ever reads `feature_capture.output` again (the injector branch reads
`adapter_injector.original_feature_5d` instead) — it's a harmless no-op hook that still
runs every step (negligible cost, since its hook body is a single tuple unpack + attribute
set), but it's dead weight worth removing or documenting as intentionally-kept-for-possible-future-use.

### 2.7 LOW — duplicated debug-snapshot recording code between the injector-enabled and injector-disabled branches

The `debug_overfit_single_batch` / `debug_overfit_pose_batch` / `debug_overfit_film_batch`
snapshot-recording blocks (`train_vq.py:460-496` and `671-696`/`585-594`) are near-identical
copies, one per code path (`if enable_continuous_adapter_injection: ... else: ...`). They
were kept in sync manually across this session's edits, but any future change to one
(e.g. adding a new snapshot field) needs the same edit made twice by hand, with no
compiler/test to catch a missed spot beyond visually diffing the two blocks. Consider
factoring the snapshot-recording into a small shared helper that takes
`(tokenizer_output, pose_output, film_output)` regardless of which branch produced them.

### 2.8 COSMETIC — stale section-number references in docstrings

`_report_debug_pose_overfit`'s docstring says "섹션 22", `_report_debug_overfit` says
"섹션 12", both `_report_debug_film_overfit` and `_report_debug_injection_overfit` say
"섹션10" (same number, from two different specs' section numbering) — these were accurate
against whichever numbered spec was being implemented turn-by-turn, but numbering has been
reused/reset across the several multi-section specs given this session. Harmless, but
confusing to a future reader trying to trace "섹션 22" back to an actual document.

### 2.9 Design note (not a bug) — single global `grad_clip` across radically different parameter-group scales

`torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)` computes ONE combined norm
across the ~1.52B-parameter UNet and the ~1-parameter `injection_scale` (and the small
tokenizer/pose/FiLM branches) together. In practice the UNet's gradient norm dominates the
combined norm almost entirely, so this global clip is effectively a no-op for the tiny
branches' own gradients (their contribution to the sum is negligible). This has been the
behavior since the tokenizer stage and never caused an observed problem (AdamW itself is
per-parameter adaptive), but it's worth knowing this is *not* per-branch clipping if a
future stage needs to protect a specific small module from a gradient spike.

## 3. What's solid (verified via actual execution, not just read-through)

- **Baseline invariance** (`test_adapter_baseline_invariance.py`, real UNet/PoseNet/VAE):
  adapter disabled vs. adapter enabled+`mode=none` vs. adapter enabled+`mode=residual,scale=0`
  all differ from each other by *exactly* the same amount as running the baseline twice in a
  row (pure GPU floating-point non-determinism, `max_abs_diff≈1.6e-2`, `loss_diff≈6.6e-7`
  in all three comparisons identically).
- **Zero-init identity invariants** at every stage: `SpatioTemporalContinuousTokenizer`
  round-trip (`test_visual_tokenizer.py`), `PoseConditionedFiLM` at `film_scale=0`
  (`test_pose_conditioned_film.py`), `ContinuousAdapterInjector` at `injection_scale=0`
  (`test_continuous_adapter_injector.py`) — all hold to `atol=1e-6`.
- **Gradient starvation was found and fixed empirically, not assumed**: double zero-init
  (`film_scale=0` + zero-init last Linear) produces *exactly* zero gradient into the FiLM
  MLP, reproduced on real project data via `debug_overfit_film_batch`, documented in
  `pose_conditioned_film.py`'s own docstring, and resolved via `film_scale_init: 1e-3`.
- **`res_samples[-1] is sample` aliasing assumption** (the crux of how residual injection
  can replace both the main path and the skip connection consistently) was verified against
  the actual installed `diffusers==0.37.0` source and the actual loaded UNet config, not
  assumed — and the injector raises a clear `RuntimeError` if this assumption ever breaks.
- **Real inference integration works end-to-end**: checkpoint's extra
  `continuous_tokenizer.*` / `pose_bundle_encoder.*` / `pose_conditioned_film.*` /
  `continuous_adapter_injector.*` keys load with `missing=[] unexpected=[]` into freshly
  constructed modules, and the auto-detecting forward pre-hook correctly handles the real
  pipeline's per-tile, per-CFG-branch varying `(batch_size, num_frames)` — after fixing the
  `tile_size` divisibility issue found on the first real run (see git history / prior
  conversation for the fix).

## 4. Test coverage summary

| File | Tests | Status |
|---|---|---|
| `test_visual_tokenizer.py` | 5 | pass |
| `test_pose_bundle_encoder.py` | 4 | pass |
| `test_pose_conditioned_film.py` | 5 | pass |
| `test_continuous_adapter_injector.py` | 5 | pass |
| `test_adapter_baseline_invariance.py` | 1 (A/A2/B/C real-model comparison) | pass |

No coverage exists yet for: the CFG-uncond-branch injection side effect (2.2), or a
regression test that would catch the `occupancy_sigmoid_temperature` default drift (2.1) —
a simple `assert PoseBundleEncoder(...).local_occupancy.sigmoid_temperature == 0.01`-style
default-value test in `train_vq.py`'s own construction path would have caught 2.1
immediately.

## 5. Suggested priority order for fixes

1. Fix 2.1 (`train_vq.py:190` default) — one-line fix, real latent risk.
2. Decide on 2.2 (CFG-uncond injection) before drawing any conclusions from
   inference-quality comparisons — this affects how "continuous_injection" results should
   be interpreted.
3. Either wire in or remove 2.3 (`pose_shuffle_diagnostic`).
4. 2.4–2.9 are lower priority, quality/maintainability items rather than correctness risks.
