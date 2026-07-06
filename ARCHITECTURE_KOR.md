# 아키텍처 — MimicMotion을 위한 Continuous Pose-Conditioned Adapter

**브랜치**: `add_tokenizer`
**최종 수정**: 2026-07-05

이 문서는 이 저장소의 MimicMotion(SVD UNet + PoseNet) 베이스라인 위에 구축된 auxiliary
adapter 파이프라인을 설명한다. 각각 config flag로 독립적으로 켜고 끌 수 있는 4단계로
구성되며, 각 단계는 비활성화했을 때 바로 이전 단계의 동작을 정확히 재현하도록 설계되었다
(단순히 코드를 읽어서가 아니라 실제로 검증됨 — `test_adapter_baseline_invariance.py` 참고).

```
enable_continuous_tokenizer          -> visual_tokenizer.py
enable_pose_bundle_encoder           -> pose_bundle_encoder.py
enable_pose_conditioned_film         -> pose_conditioned_film.py     (위 두 개 필요)
enable_continuous_adapter_injection  -> continuous_adapter_injector.py (위 세 개 모두 필요)
```

## 1. 베이스라인 (변경 없음)

- **UNet**: `mimicmotion/modules/unet.py::UNetSpatioTemporalConditionModel` — SVD UNet,
  down_blocks 4개(`block_out_channels=[320,640,1280,1280]`), EDM/v-prediction
  preconditioning으로 학습됨(`train_vq.py`의 `c_skip/c_out/c_in/loss_weight`가
  `EulerDiscreteScheduler`의 실제 v-prediction 공식과 일치함 — `c_out`이 음수임에 주의).
- **PoseNet**: `mimicmotion/modules/pose_net.py::PoseNet` — `conv_layers`(→128채널) 다음
  `final_proj`(1x1 conv, 128→320채널)로, 그 출력이 UNet의 `conv_in`에 더해진다.
- 두 모듈 모두 이전과 정확히 동일하게 diffusion loss로 직접 학습된다. 위 4개 flag가 모두
  `false`일 때는 이 문서의 어떤 내용도 그 경로를 바꾸지 않는다.

## 2. 1단계 — Visual Tokenizer (`visual_tokenizer.py`)

선택된 UNet feature를 겹치지 않는 spatio-temporal patch token으로 변환했다가 다시
복원하는, **continuous**(비-VQ) bottleneck.

- **선택된 feature**: `down_blocks[1]`의 출력(`res_samples`가 아니라 `sample`), 실측 shape은
  `resolution=576, num_frames=8` 기준 `(B*T, 640, 32, 18)` (추측이 아니라 hook으로 확인됨).
- **`UNetFeatureCapture`**: 범용 상시 forward hook(이 feature와 `PoseNet.conv_layers` 양쪽에
  재사용됨), 호출마다 `.output` 참조를 저장하며 출력을 절대 변경하지 않는다.
- **`SpatioTemporalContinuousTokenizer`**:
  - `patchify`: `[B,T,C,H,W] → [B,Nt,Nh,Nw, Lt·Ph·Pw·C]` (`Nt=T/Lt, Nh=H/Ph, Nw=W/Pw`)
  - `encode`: `Linear(patch_dim → embed_dim)` (LayerNorm + Linear) → `visual_tokens`
  - `decode`: `Linear(embed_dim → patch_dim)`
  - `unpatchify`: `patchify`의 역연산 (정확한 permute/reshape, round-trip을 `atol=1e-6`까지
    검증함)
  - 기본 patch grid: `temporal_patch_size=4, spatial_patch_height=4, spatial_patch_width=3`
    → `patch_dim=30720`, `embed_dim=64` (약 480배 압축 bottleneck)
- **Loss**: `smooth_l1_loss(reconstructed, original_feature)`, 가중치는
  `lambda_tokenizer_reconstruction` (adapter-injection 단계 기준 `0.1`; adapter가
  diffusion_loss에 영향을 줄 수 있게 되기 전에는 `1.0`이었음).
- 이 단계 자체만으로는 UNet에 주입되지 않는다 — 주입 시점은 4단계 참고.

## 3. 2단계 — Pose Bundle Encoder (`pose_bundle_encoder.py`)

pose branch 자체의 feature를 visual feature의 spatio-temporal grid에 정렬시키고,
visual branch와 완전히 병렬로 bundle당 두 개의 pose 신호를 생성한다.

- **선택된 feature**: `PoseNet.conv_layers`의 출력(`final_proj` 이전), 실측 shape은
  `(B*T, 128, 128, 72)`.
- **`PoseFeatureAligner`**: visual feature의 `(H,W)`로 bilinear interpolation하며, 선택적
  정규화(`none` / `channel_layernorm` / `channel_standardize`; `none`이 검증된
  기본값 — occupancy는 raw activation 크기가 필요하기 때문).
- **`GlobalPoseEncoder`**: 각 temporal bundle을 mean-pool한 뒤
  `LayerNorm→Linear→SiLU→Linear` → `global_pose_embedding [B,Nt,pose_embed_dim]`
  (기본값 `pose_embed_dim=32`).
- **`PoseReconstructionHead`**: `global_pose_embedding → 복원된 pooled pose feature`로,
  self-supervised reconstruction loss(target은 detach됨)를 통해 `GlobalPoseEncoder`를
  학습시킨다 — FiLM 이전 단계에서 global encoder를 학습시키는 *유일한* 신호다.
- **`LocalPoseOccupancy`**: deterministic이며 학습 파라미터가 없다. 채널에 대한
  `abs_mean` activation → patch-pool → `sigmoid((x - spatial_mean) / temperature)`.
  `occupancy_sigmoid_temperature=0.01` (**1.0이 아님** — 실측 결과 `1.0`은 occupancy를
  `occ_std≈0.002, active_ratio≈0`으로 붕괴시켰고, `0.01`은 `occ_std≈0.18,
  active_ratio≈0.26`을 준다. 이 값의 코드 fallback 기본값 중 하나에 남아있는 잠재적
  불일치는 `CODE_REVIEW.md` §2.1 참고.)
- **Loss**: `pose_reconstruction_loss` (self-supervised), 가중치는
  `lambda_pose_reconstruction` (`0.1`).
- Collapse 진단(`global_pose_embedding_stats`, `local_occupancy_stats`)이 std≈0,
  occupancy≈all-0/all-1, temporal variance 0 등에 대해 경고를 낸다.

## 4. 3단계 — Pose-Conditioned FiLM (`pose_conditioned_film.py`)

pose 정보가 실제로 visual token을 변조하는 첫 단계다(이 단계 자체만으로는 여전히 UNet을
건드리지 않는다).

```
modulated_tokens = visual_tokens
    + film_scale * local_pose_occupancy * (gamma * visual_tokens + beta)
gamma, beta = film_mlp(global_pose_embedding).chunk(2, dim=-1)
```

- `film_mlp`: `LayerNorm → Linear → SiLU → Linear(→ 2·visual_embed_dim)`, 기본값으로
  마지막 Linear는 zero-init된다.
- `film_scale`: 학습 가능한 스칼라, 기본값 `0.0` 초기화 → step 0에서
  `modulated_tokens == visual_tokens`가 정확히 성립함(`atol=1e-6`, 검증됨).
- **알려져 있고 문서화된 gradient-starvation 함정**: `film_scale_init=0.0`**과**
  `film_last_linear_init="zero"`를 함께 사용하면(둘 다 기본값), `film_scale`과
  `film_mlp`로 흘러가는 gradient가 **정확히 0**이 된다(단순히 작은 게 아니라) — 실제
  프로젝트 데이터로 재현됨. 이번 세션에서 사용한 수정: `film_scale_init: 1.0e-3`
  (초기 identity 상태는 유지하면서 gradient 막힘만 해제) — 전체 유도 과정은 모듈 자체의
  docstring 참고.
- Loss: `film_reg_loss = gamma.pow(2).mean() + beta.pow(2).mean()`, 가중치는
  `lambda_film_reg` (`1e-4`).
- `pose_film_shuffle_diagnostic`: pose 신호의 temporal-bundle 순서를 섞은 뒤
  `modulated_tokens`/reconstruction이 실제로 달라지는지 확인한다(logging 전용이며 loss에는
  절대 들어가지 않음). `randperm` 대신 cyclic shift를 사용하는데 — 이 프로젝트의 실제
  config처럼 `Nt`가 2만큼 작을 때는 random permutation이 50% 확률로 identity가 되어버리는
  현상이 실제로 관찰되어 수정한 것이다.

## 5. 4단계 — Continuous Adapter Residual Injection (`continuous_adapter_injector.py`)

`reconstructed_feature`가 UNet의 실제 forward pass로 다시 흘러 들어가는 첫 단계이며,
따라서 `diffusion_loss` 자체가 adapter의 영향을 받을 수 있게 되는 첫 단계이기도 하다.

```
feature_delta      = reconstructed_feature - visual_feature
feature_for_unet    = visual_feature + injection_scale * feature_delta   # mode="residual"
                    = visual_feature                                     # mode="none"
```

- `injection_scale`: 학습 가능한 스칼라, `0.0` 초기화 → step 0에서 베이스라인과 정확히
  동일.
- **메커니즘 — UNet 코드 수정이 아니라 forward hook.** `down_blocks[1]`은
  `(sample, res_samples)`를 반환하는데, 이 block에 downsampler가 있기 때문에
  `res_samples[-1]`이 곧 `sample`이다(같은 tensor 객체 — 실제 설치된
  `diffusers==0.37.0` 소스와 실제로 로드된 모델 config로 확인됨, 추측이 아님).
  따라서 `down_blocks[1]`에 건 forward hook은 새로운
  `(feature_for_unet, res_samples[:-1] + (feature_for_unet,))` 튜플을 반환함으로써
  이어지는 main path와 그것과 aliasing된 skip-connection 항목을 *동시에* 일관되게
  교체할 수 있다 — PyTorch는 hook이 `None`이 아닌 값을 반환하면 모듈의 실제 출력을 그
  값으로 대체한다. 이 aliasing 가정이 (예: 다른 base UNet으로 바뀌는 등) 깨지면
  injector는 조용히 잘못 동작하는 대신 `RuntimeError`를 던진다.
- **tokenizer→pose→FiLM 파이프라인 전체가 이 하나의 hook 안에서 실행된다.** 주입이
  `model_pred`를 만들어내는 바로 그 `unet(...)` 호출 *도중에* 일어나야 하기 때문이며,
  앞 단계들처럼 training loop에서 나중에 계산할 수 없다.
- **배치/프레임 자동 감지**: 최상위 `unet`에 걸린 별도의 forward *pre*-hook이 매 호출마다
  들어오는 5D `sample` tensor에서 직접 `(batch_size, num_frames)`를 읽어온다. 이것이
  중요한 이유는 `mimicmotion/pipelines/pipeline_mimicmotion.py`의 실제 추론 루프가
  학습 때와 전혀 다르게 UNet을 호출하기 때문이다: tile마다 CFG branch별로 한 번씩
  (`batch_size=1`), `num_frames`는 tile마다 달라진다 — pre-hook 덕분에 호출자가 이를
  따로 추적할 필요가 없어진다.
- Loss: `injection_reg_loss = injection_scale.pow(2)`, 가중치는
  `lambda_injection_reg` (`1e-4`), `adapter_injection_mode="residual"`일 때만
  `total_loss`에 더해진다.
- **알려진 미해결 이슈** (`CODE_REVIEW.md` §2.2 참고): 추론 시 CFG-unconditional branch도
  injection을 받는다(tile당 두 UNet 호출이 동일한 pose feature를 공유하기 때문). 이번
  세션의 어떤 스펙에서도 이를 다루지 않았으므로, inference-quality 비교로부터 강한
  결론을 내리기 전에 결정이 필요하다.

## 6. 데이터 흐름 (training loop, `enable_continuous_adapter_injection=true`)

```
pose_images ──> encode_pose_latents (PoseNet) ──> pose_latents ──┐  (pose_feature_capture.output을 캡처함)
                                                                    │
video_latents, ref_embeds, sigma, noise ──> latent_input ─────────┤
                                                                    ▼
                                                          comps.unet(latent_input, ..., pose_latents=pose_latents)
                                                                    │
                                                     down_blocks[0] │
                                                                    ▼
                                              ┌── down_blocks[1] ──┴─────────────────────────┐
                                              │   sample, res_samples = block(...)            │
                                              │   [ContinuousAdapterInjector hook 발동]        │
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
              + lambda_injection_reg            * injection_reg_loss   (mode="residual"일 때만)
```

`enable_continuous_adapter_injection=false`일 때는 동일한 네 부분(tokenizer, pose bundle
encoder, FiLM)이 `model_pred`가 생성된 *이후* training loop에서 수동으로 계산되며,
(주입하지 않는) 순수 `UNetFeatureCapture` hook에서 값을 읽어온다 — 이 모드에서는 UNet에
아무것도 다시 흘러 들어가지 않으므로 위치만 다를 뿐 수치적으로는 동일한 결과다.

## 7. Config flag (in `configs/train_config.yaml`)

| Flag | 필요 조건 | `true`일 때 효과 |
|---|---|---|
| `enable_continuous_tokenizer` | — | patchify/encode/decode/unpatchify auxiliary branch |
| `enable_pose_bundle_encoder` | — | global pose embedding + local occupancy auxiliary branch |
| `enable_pose_conditioned_film` | tokenizer + pose_bundle | decode 전에 FiLM으로 token 변조 |
| `enable_continuous_adapter_injection` | tokenizer + pose_bundle + FiLM | UNet forward에 residual 주입(`adapter_injection_mode` 참고) |
| `adapter_injection_mode` | injection 활성화됨 | `"none"`(계산만) 또는 `"residual"`(실제로 주입) |

각 단계 고유의 learning rate는 **필수**(fallback 없음) config key이며, 각자 별도의
optimizer param group을 가진다: `tokenizer_learning_rate`, `pose_encoder_learning_rate`,
`film_learning_rate`, `injection_learning_rate`.

## 8. 체크포인트 포맷

`train_vq.py`의 `_trainable_state_dict()`가 저장하는 단일 flat `state_dict`:

```
unet.*                          (항상)
pose_net.*                      (항상)
continuous_tokenizer.*          (enable_continuous_tokenizer일 때)
pose_bundle_encoder.*           (enable_pose_bundle_encoder일 때)
pose_conditioned_film.*         (enable_pose_conditioned_film일 때)
continuous_adapter_injector.*   (enable_continuous_adapter_injection일 때 — `injection_scale` 하나뿐)
```

기존 `mimicmotion/utils/loader.py::create_pipeline`(`inference.py` / `inference_trained.py`가
사용함)는 이를 `strict=False`로 로드하므로, 추가된 prefix key들은 조용히 무시되고
("Unexpected key"로 로그만 남음) 학습 시 어떤 단계를 활성화했든 상관없이 순수 베이스라인
파이프라인으로 추론이 동작한다. 추론에서 adapter를 실제로 *작동*시키려면(단순히 학습만
하는 게 아니라), 별도 스크립트인 `inference_with_adapter.py`가 그 추가 key들로부터 네
모듈을 재구성하고 로드된 파이프라인의 `unet`/`pose_net`에 `ContinuousAdapterInjector`를
다시 붙인다.

## 9. 파일 맵

| 파일 | 역할 |
|---|---|
| `visual_tokenizer.py` | 1단계 모듈 + `UNetFeatureCapture` + B↔BT reshape 헬퍼 |
| `pose_bundle_encoder.py` | 2단계 모듈 + collapse 진단 + `pose_shuffle_diagnostic`(현재 dead code, `CODE_REVIEW.md` §2.3 참고) |
| `pose_conditioned_film.py` | 3단계 모듈 + `pose_film_shuffle_diagnostic` |
| `continuous_adapter_injector.py` | 4단계 모듈(hook 기반 injector) |
| `train_vq.py` | 4단계 전체를 엮는 training loop + config validation cascade |
| `inference_with_adapter.py` | 실제 추론을 위해 adapter를 재구성하고 다시 붙임 |
| `test_visual_tokenizer.py`, `test_pose_bundle_encoder.py`, `test_pose_conditioned_film.py`, `test_continuous_adapter_injector.py` | 독립 unit test, 테스트 대상 모듈 외 프로젝트 의존성 없음 |
| `test_adapter_baseline_invariance.py` | 실제 모델 기반 A/B/C 베이스라인-불변성 검사 |
| `CODE_REVIEW.md` | 이 파이프라인에 대한 이슈 트래커 / 리뷰 노트 |
| `ARCHITECTURE.md` | 이 문서(영문 원본) |

## 10. 검증 로그

**2026-07-05**: 위의 모든 주장을 (이 문서로부터 다시 유추한 게 아니라) 실제 소스와 다시
대조 확인함 — `unet.py`의 L434/442-444/450-465/502, `train_vq.py`의 config-flag 연결
로직과 190번째 줄에 남아있는 오래된 `occupancy_sigmoid_temperature=1.0` fallback(여전히
존재함, `CODE_REVIEW.md` §2.1 참고), `configs/train_config.yaml`의 실제 값들,
`pipeline_mimicmotion.py`의 이중 `self.unet(...)` CFG 호출까지 확인했다. 4개의 가벼운
test suite(`test_visual_tokenizer.py`, `test_pose_bundle_encoder.py`,
`test_pose_conditioned_film.py`, `test_continuous_adapter_injector.py`, 총 19개 테스트)를
실제로 실행했으며(실행 중이던 job들과 GPU를 두고 경합하지 않도록 CPU-only로 실행) 모두
통과했다. `test_continuous_adapter_injector.py`가 출력한 실제 feature shape
`(8, 640, 32, 18)`도 이 문서의 주장과 일치한다. `test_adapter_baseline_invariance.py`는
(실제 UNet/VAE 가중치와 GPU가 필요해서) 이번에 다시 실행하지 않았고 기존 기록을 그대로
채택했다. 불일치는 발견되지 않았으며, 내용 변경은 필요하지 않았다.
