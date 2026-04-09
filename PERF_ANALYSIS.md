# MimicMotion Performance Optimization Analysis

## Pipeline Overview (for reference)

```
preprocess()           ~5-30s depending on video length
  ├─ DWPose detection  (sequential per-frame, ONNX)
  ├─ Pose rescaling    (numpy loops)
  └─ Pose drawing      (cv2, upscaled canvas per frame)

run_pipeline()         ~60-300s (dominates runtime)
  ├─ CLIP encoding     (single image, fast)
  ├─ VAE encoding      (single image, fast)
  ├─ Denoising loop    (25 steps x N tiles — the bottleneck)
  │   ├─ PoseNet       (per tile)
  │   └─ UNet x2       (CFG: unconditional + conditional)
  ├─ VAE decoding      (chunked, 8 frames)
  └─ tensor2vid        (postprocessing)
```

---

## Findings

### P0 — HOT PATH: Denoising Loop Inefficiencies

#### 1. Redundant `torch.cat` for CFG on every timestep
**File:** `pipeline_mimicmotion.py:555,559`
```python
for i, t in enumerate(timesteps):            # 25 iterations
    latent_model_input = torch.cat([latents] * 2)                          # line 555
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)  # line 559
```
Each timestep allocates two new tensors via `torch.cat`. With 25 steps, that's 50 allocations of full-latent-sized tensors.

**Optimization:** Pre-allocate `latent_model_input` once before the loop. On each step, write into slices instead of allocating:
```python
latent_model_input = torch.empty(...)  # pre-allocate once
for i, t in enumerate(timesteps):
    latent_model_input[:1, :, :4] = latents
    latent_model_input[1:, :, :4] = latents
    latent_model_input[:, :, 4:] = image_latents  # already static across steps
```
`image_latents` is constant across all timesteps — concatenating it every iteration is pure waste.

**Impact:** Eliminates ~50 large GPU allocations per inference. Reduces memory pressure and GC overhead.

---

#### 2. Host-to-device transfer inside inner loop (pose data)
**File:** `pipeline_mimicmotion.py:569`
```python
for idx in indices:                          # N tiles per timestep
    pose_latents = self.pose_net(image_pose[idx].to(device))
```
`image_pose` lives on CPU. Every tile, for every timestep, a slice is transferred to GPU. For 25 steps x 5 tiles = 125 H2D transfers of the same data.

**Optimization:** Move `image_pose` to device once before the loop and keep it there. Or pre-compute all PoseNet outputs (they don't depend on timestep):
```python
# Before denoising loop:
image_pose = image_pose.to(device)
pose_cache = {tuple(idx): self.pose_net(image_pose[idx]) for idx in indices}
# Inside loop:
pose_latents = pose_cache[tuple(idx)]
```

**Impact:** Eliminates 125+ redundant H2D transfers and 125+ redundant PoseNet forward passes (PoseNet output is deterministic and timestep-independent). This alone could save seconds per inference.

---

#### 3. Two separate UNet forward passes instead of batched
**File:** `pipeline_mimicmotion.py:570-590`
```python
_noise_pred = self.unet(latent_model_input[:1, idx], ..., pose_latents=None, ...)
_noise_pred = self.unet(latent_model_input[1:, idx], ..., pose_latents=pose_latents, ...)
```
CFG requires two UNet passes (unconditional + conditional). These are done sequentially despite being independent. The split is because `pose_latents=None` for unconditional vs `pose_latents=pose_latents` for conditional.

**Optimization:** Batch both into a single UNet call by concatenating inputs and passing `pose_latents` with zeros for the unconditional path:
```python
pose_input = torch.cat([torch.zeros_like(pose_latents), pose_latents])
_noise_pred = self.unet(
    latent_model_input[:, idx], t,
    encoder_hidden_states=image_embeddings,
    added_time_ids=added_time_ids,
    pose_latents=pose_input,
    ...
)[0]
```
This requires verifying that `pose_latents=None` produces identical results to `pose_latents=zeros` in the UNet. If the UNet adds pose_latents (common pattern), zeros would be equivalent.

**Impact:** ~2x UNet throughput (single batched pass vs two sequential). This is the single highest-impact optimization since UNet dominates runtime. Depends on VRAM — requires ~2x peak UNet memory but saves wall time.

---

#### 4. `noise_pred` and `noise_pred_cnt` re-allocated every timestep
**File:** `pipeline_mimicmotion.py:562-563`
```python
noise_pred = torch.zeros_like(image_latents)
noise_pred_cnt = image_latents.new_zeros((num_frames,))
```
These are allocated fresh inside the timestep loop. They could be allocated once and `.zero_()`'d.

**Impact:** Minor — 25 allocations saved. But reduces GC pressure.

---

### P1 — PREPROCESSING: Sequential DWPose Detection

#### 5. Frame-by-frame pose detection in Python loop
**File:** `dwpose/preprocess.py:37`
```python
detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
```
Each frame goes through YOLOX detection + DWPose estimation sequentially. For a 72-frame video at stride 2, that's ~36 sequential ONNX inference calls.

**Optimization options:**
- **Batch ONNX inference:** If the ONNX models support dynamic batch dimensions, batch multiple frames per call. YOLOX and DWPose ONNX models often support batch>1.
- **Parallel frame processing:** Use `concurrent.futures.ThreadPoolExecutor` to overlap CPU preprocessing with ONNX GPU inference (if using CUDA EP).
- **Cache pose results:** If the same video is used multiple times, cache detected poses to disk.

**Impact:** Could reduce preprocessing from ~10-30s to ~3-10s depending on batch size support.

---

#### 6. Per-frame pose drawing with upscaled canvas
**File:** `dwpose/util.py:100-133`
```python
def draw_pose(pose, H, W, ref_w=2160):
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    # ... draw on upscaled canvas ...
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
```
Called in a loop from `preprocess.py:56`:
```python
for detected_pose in detected_poses:
    im = draw_pose(detected_pose, height, width)
    output_pose.append(np.array(im))
```
Each frame allocates a large canvas (e.g. 2160x3840x3 = ~25MB), draws on it, then resizes back down. For 36 frames that's ~900MB of temporary allocations cycled through.

**Optimization:**
- Reduce `ref_w` if quality is acceptable at lower internal resolution (e.g. 1080 instead of 2160 = 4x less memory, faster drawing).
- Pre-allocate the canvas once and `.fill(0)` between frames instead of re-allocating.
- Consider whether the upscale-then-downscale is even needed: if final output is 576px, drawing at 1080 or even 576 directly may be sufficient.

**Impact:** Moderate. Saves ~seconds of preprocessing and reduces memory churn.

---

#### 7. Unnecessary `.copy()` on numpy arrays
**File:** `inference.py:61`
```python
return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, ...
```
`pose_pixels` is already a freshly-stacked numpy array from `np.stack(output_pose)` at `preprocess.py:58`. The `.copy()` is redundant — `np.stack` already returns a new contiguous array.

Also in `dwpose_detector.py:34`:
```python
oriImg = oriImg.copy()  # unnecessary if not mutated upstream
```

**Impact:** Minor — saves one full-frame array copy per inference.

---

### P1 — POSTPROCESSING

#### 8. Redundant image format round-trip
**File:** `inference.py:65` → `pipeline_mimicmotion.py:126-127`
```python
# inference.py:65 — tensor [-1,1] → PIL [0,255]
image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]

# pipeline_mimicmotion.py:126-127 — PIL → numpy → tensor → normalize back to [-1,1]
image = self.image_processor.pil_to_numpy(image)
image = self.image_processor.numpy_to_pt(image)
image = image * 2.0 - 1.0
```
The pipeline already normalized the image to `[-1, 1]`. Then `run_pipeline` converts it back to PIL (losing precision via uint8 quantization). Then `_encode_image` converts it back to tensor and re-normalizes. This is:
1. Lossy (uint8 quantization)
2. Wasteful (3 unnecessary conversions)

**Optimization:** Keep the image as a float tensor throughout. Pass the tensor directly to `_encode_image` (it already has a `isinstance(image, torch.Tensor)` fast path at line 125).

**Impact:** Eliminates 3 conversion steps and avoids uint8 quantization loss. Could marginally improve output quality.

---

#### 9. VAE decode: immediate `.cpu()` per chunk prevents decode overlap
**File:** `pipeline_mimicmotion.py:240`
```python
frame = self.vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
frames.append(frame.cpu())  # sync + transfer per chunk
```
Each `.cpu()` is a synchronous D2H transfer that blocks before the next VAE decode can start.

**Optimization:** Use pinned memory + async transfer:
```python
frames.append(frame.to('cpu', non_blocking=True))
# ... after loop:
torch.cuda.synchronize()
```
Or decode all on GPU if VRAM allows, then transfer to CPU once.

**Impact:** Modest. Overlaps D2H transfer with next decode chunk computation.

---

### P2 — MODEL LOADING / DEVICE MANAGEMENT

#### 10. Sequential model CPU offloading adds latency
**File:** `pipeline_mimicmotion.py:471-492,546-547,613-617`
```python
self.image_encoder.to(device)    # H2D
image_embeddings = self._encode_image(...)
self.image_encoder.cpu()         # D2H

self.vae.to(device)              # H2D
image_latents = self._encode_vae_image(...)
self.vae.cpu()                   # D2H

self.pose_net.to(device)         # H2D
self.unet.to(device)             # H2D
# ... denoising loop ...
self.pose_net.cpu()              # D2H
self.unet.cpu()                  # D2H

self.vae.decoder.to(device)      # H2D
frames = self.decode_latents(...)
```
Each `.to(device)` and `.cpu()` is a full model parameter transfer. The image encoder (~400M params in fp16 = ~800MB) and VAE encoder are transferred to GPU, used once, then moved back.

**Optimization:**
- If VRAM allows, keep all models on GPU (skip CPU offloading entirely). This is the simplest win if VRAM >= 24GB.
- If VRAM is tight, use CUDA streams to overlap model loading with computation.
- At minimum, keep VAE decoder on GPU during the whole call since it's needed at the end anyway.

**Impact:** Could save 2-5 seconds of model transfer overhead per inference.

---

### P2 — VECTORIZATION

#### 11. Nested Python loops in DWPose post-processing
**File:** `dwpose_detector.py:44-49`
```python
for i in range(len(subset)):
    for j in range(len(subset[i])):
        if subset[i][j] > 0.3:
            subset[i][j] = int(18 * i + j)
        else:
            subset[i][j] = -1
```
This nested loop can be vectorized with numpy:
```python
mask = subset > 0.3
indices = np.arange(18)[None, :] + 18 * np.arange(len(subset))[:, None]
subset = np.where(mask, indices, -1).astype(float)
```

**Impact:** Negligible in isolation, but pattern repeats in drawing code.

---

#### 12. Pose drawing loops (body, hands, face) are serial Python
**File:** `dwpose/util.py:29-55,65-86,90-97`

All drawing operations use nested Python loops iterating over keypoints and drawing with cv2. This is inherently serial and CPU-bound.

**Optimization:** Not easily vectorizable due to cv2 draw calls. Could potentially:
- Use vectorized numpy operations for computing polygon coordinates, then batch cv2 calls.
- For the face points (68-92 keypoints), batch the `cv2.circle` calls with pre-computed parameters.
- These loops are unlikely to be the bottleneck vs. ONNX inference.

**Impact:** Low. Preprocessing is dominated by ONNX inference, not drawing.

---

### P3 — MINOR / SPECULATIVE

#### 13. `torch.compile` / CUDA Graphs for UNet
The UNet is called `25 * N_tiles * 2` times with the same tensor shapes. This is an ideal candidate for `torch.compile` (PyTorch 2.x) or CUDA graphs. The current code does nothing to leverage these.

**Optimization:** Wrap the UNet in `torch.compile(mode="reduce-overhead")` or capture a CUDA graph of the forward pass.

**Impact:** Potentially 20-40% speedup on the denoising loop (architecture-dependent).

---

#### 14. PoseNet could use `torch.compile`
PoseNet is a simple 7-layer CNN called `N_tiles * 25` times (or once per tile if cached per finding #2).

**Impact:** Negligible if finding #2 is implemented (PoseNet would only run N_tiles times total).

---

#### 15. `tensor2vid` iterates over batch dimension
**File:** `pipeline_mimicmotion.py:37-47`
```python
for batch_idx in range(batch_size):
    batch_vid = video[batch_idx].permute(1, 0, 2, 3)
    batch_output = processor.postprocess(batch_vid, output_type)
    outputs.append(batch_output)
```
Batch size is typically 1 in inference, so this is a non-issue in practice.

**Impact:** None for typical usage.

---

#### 16. Scheduler `.step()` not profiled
**File:** `pipeline_mimicmotion.py:603`
```python
latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
```
Euler scheduler step is lightweight (element-wise ops), but `return_dict=False` is already used which avoids dict overhead. No issue here.

---

## Priority Summary

| # | Finding | Impact | Effort | Risk |
|---|---------|--------|--------|------|
| **2** | Cache PoseNet outputs (timestep-independent) | **High** | Low | None — deterministic |
| **3** | Batch CFG UNet passes | **Very High** | Medium | Must verify pose_latents=zeros == None |
| **1** | Pre-allocate latent_model_input, avoid repeated cat | **Medium-High** | Low | None |
| **8** | Eliminate tensor→PIL→tensor round-trip | **Medium** | Low | Must verify CLIP encoding path |
| **5** | Batch DWPose ONNX inference | **Medium** | Medium | Depends on ONNX model support |
| **9** | Async VAE decode D2H transfers | **Medium** | Low | None |
| **10** | ~~Skip CPU offloading if VRAM allows~~ | ~~**Medium**~~ | ~~Low~~ | **DONE** — skipped on MPS |
| **13** | `torch.compile` for UNet | **High** | Low-Medium | May need graph breaks debugging |
| **6** | Reduce pose drawing canvas size | **Low-Medium** | Low | May reduce pose image quality |
| **4** | Pre-allocate noise_pred buffers | **Low** | Low | None |
| **7** | Remove unnecessary `.copy()` | **Low** | Trivial | None |
| **11** | Vectorize subset indexing loop | **Low** | Low | None |

## Recommended Implementation Order

1. **#2 — Cache PoseNet outputs** (trivial, guaranteed safe, high impact)
2. **#1 — Pre-allocate latent_model_input** (easy, safe, reduces memory churn)
3. **#8 — Eliminate format round-trip** (easy, improves quality + speed)
4. **#3 — Batch CFG UNet passes** (highest potential impact, needs verification)
5. **#13 — torch.compile** (potentially large win, may need debugging)
6. **#5 — Batch DWPose** (preprocessing speedup)
7. Everything else

## Implemented Optimizations & Benchmarks

### Changes made (branch: `add-mps-support`)

1. **`assign=True` in `load_state_dict`** — avoids copying checkpoint tensors into model params; directly swaps references. Loading RSS: 6.04 GB → 3.89 GB (**-2.15 GB**).
2. **`del checkpoint` + `del mimicmotion_models`** — frees checkpoint dict and wrapper immediately after use.
3. **Skip CPU offloading on MPS** — on unified memory, `.cpu()` / `.to(device)` round-trips are no-ops that waste time and can double memory by creating separate CPU and MPS copies of the same tensors.

### Benchmark: 576p on Apple M3 Max (48 GB unified)

This resolution previously **OOMed** ("MPS allocated: 6.20 GiB, other allocations: 57.22 GiB, max allowed: 63.65 GiB").

**Config:** 576x1024, 18 total frames, tile_size=16, overlap=6, 5 denoise steps, fp16, seed=42

| Stage | Time |
|-------|------|
| Model loading | 5.6s |
| Preprocessing (DWPose, 17 frames, CoreML) | 8.3s |
| Pipeline (denoising + VAE decode) | 1714.0s |
| Video save | 0.1s |
| **Total wall time** | **1729.7s (28.8 min)** |

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| Loading RSS peak | 6.04 GB | 3.89 GB |
| 576p MPS peak | OOM (>63.65 GB) | **43.04 GB** |
| MPS allocated (post-run) | — | 4.22 GB |

### Benchmark: 256p minimal (fast iteration config)

**Config:** 256x448, 18 total frames, tile_size=8, overlap=4, 2 denoise steps, fp16, stride=64

| Stage | Time |
|-------|------|
| Model loading | ~4s |
| Preprocessing (DWPose, 17 frames) | <1s |
| Pipeline (denoising + VAE decode) | ~17s |
| **Total wall time** | **~28s** |

| Metric | Value |
|--------|-------|
| MPS driver peak | 12.27 GB |
| MPS allocated (post-run) | 4.46 GB |

### Hardware

- Apple M3 Max, 48 GB unified memory
- PyTorch 2.11.0, MPS backend
- ONNX Runtime with CoreMLExecutionProvider for DWPose

---

## Verification Strategy

All optimizations should produce **bit-identical outputs** (or within fp16 tolerance) given the same seed. Verification:
```python
# Before optimization
torch.manual_seed(42); output_before = run_pipeline(...)

# After optimization  
torch.manual_seed(42); output_after = run_pipeline(...)

assert torch.allclose(output_before, output_after, atol=1e-3)
```
Exception: Finding #8 (eliminating uint8 round-trip) will produce **slightly different** but **higher quality** outputs due to avoiding quantization loss. This is a strict improvement.
