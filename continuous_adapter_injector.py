"""
continuous_adapter_injector.py

visual tokenizer -> pose bundle encoder -> pose-conditioned FiLM 파이프라인을
UNet의 실제 forward pass 도중(선택된 down_block이 반환되는 순간) 실행하고,
그 결과(reconstructed_feature)를 residual로 UNet이 실제로 사용하는 feature에
주입하는 module. 이번 단계에서 처음으로 diffusion loss가 adapter의 영향을
받을 수 있게 된다.

VQ/codebook/nearest-code selection/straight-through estimator/commitment
loss/hard-soft quantization은 구현하지 않는다.

왜 forward hook인가 (추측 아님, 실제 설치된 diffusers 소스와 실제 모델
config로 확인):
    mimicmotion/modules/unet.py 의 forward()는 다음과 같다
    (L450-465, 이번 세션 이전에 실제로 읽어 확인한 내용):
        sample, res_samples = downsample_block(hidden_states=sample, ...)
        down_block_res_samples += res_samples
    즉 각 down_block은 (다음 down_block/mid_block으로 이어지는) `sample`과
    (up_blocks의 skip connection에 쓰이는) `res_samples` 튜플을 함께
    반환한다.

    이 프로젝트가 실제로 로드하는 UNet(models/stable-video-diffusion-img2vid
    -xt-1-1/unet)에서 down_blocks[1]은 `has_downsample=True`다(직접 로드해서
    `down_blocks[1].downsamplers is not None`으로 확인, 추측 아님). 설치된
    diffusers==0.37.0의
    `diffusers.models.unets.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal
    .forward()` 실제 소스(`inspect.getsource`로 직접 읽음)를 보면:
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        return hidden_states, output_states
    즉 downsampler가 있으면 반환되는 `hidden_states`(=sample)와
    `output_states`(=res_samples)의 마지막 원소가 "같은 텐서 객체"다. 따라서
    down_blocks[1]의 selected feature는 (a) 다음 down_blocks[2]로 이어지는
    `sample`과 (b) up_blocks skip connection에 쓰이는 `res_samples[-1]`
    두 곳에 동시에(같은 참조로) 나타난다. 이 hook은 두 곳을 항상 같이
    교체하며(다른 down_block의 res_sample은 절대 건드리지 않음), 만약 이
    가정이 (예: 향후 다른 base 모델로 교체돼) 깨지면 `res_samples[-1] is
    sample`이 아니게 되어 즉시 명확한 RuntimeError로 실패한다(조용히 잘못된
    동작을 하지 않음).

    forward hook이 non-None을 반환하면 PyTorch가 모듈의 실제 반환값을 그
    값으로 치환한다 -- 직접 실험으로 확인(여러 hook이 체이닝될 때 뒤 hook은
    앞 hook이 수정한 output을 보고, gradient도 정상적으로 주입된
    텐서/파라미터까지 역전파된다). unet.py 자체는 전혀 수정하지 않으므로
    (a) in-place 연산이 없고 (b) down_blocks[1] 내부의 resnet에만 적용되는
    gradient checkpointing(`self._gradient_checkpointing_func(resnet, ...)`,
    모듈 자체가 아니라 그 내부 서브모듈에 적용됨)과 충돌하지 않는다.

    pose_feature_capture.output은 이 hook이 실행되는 시점에 이미 채워져
    있어야 한다: train_vq.py의 학습 루프에서 `encode_pose_latents(...)`
    (pose_net.forward()를 내부에서 호출, pose_feature_capture 훅을 그
    순간 발동시킴)가 `comps.unet(...)` 호출보다 먼저 실행되므로, 이
    down_blocks[1] hook(=unet 내부에서 나중에 발동)이 실행될 때는 이미
    pose feature가 캡처되어 있다.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from visual_tokenizer import flatten_video_dimension, is_rank0, restore_video_dimension


class ContinuousAdapterInjector(nn.Module):
    """선택된 down_block에 forward hook을 등록해, tokenizer -> pose bundle
    encoder -> FiLM -> decode/unpatchify 파이프라인 전체를 UNet forward 도중
    실행하고, `injection_mode="residual"`일 때만 그 결과를 실제로 주입한다.

    `injection_mode="none"`이면 파이프라인은 계산되지만(로스/로깅용) UNet의
    실제 출력은 전혀 바뀌지 않는다(hook이 None을 반환). `injection_scale=0`
    으로 초기화되어 있으면 `injection_mode="residual"`이어도 초기 상태에서
    feature_for_unet은 원본 feature와 정확히 같다(atol=1e-6).

    `visual_tokenizer`/`pose_bundle_encoder`/`pose_conditioned_film`은
    `object.__setattr__`로 저장해 nn.Module의 자동 서브모듈 등록을 피한다 --
    이 세 모듈의 파라미터는 이미 train_vq.py에서 각자 별도의 optimizer param
    group으로 관리되므로, 여기서 다시 등록하면 (해롭지는 않지만) 중복이다.
    이 클래스 자신의 학습 파라미터는 `injection_scale` 하나뿐이다.
    """

    def __init__(
        self,
        unet: nn.Module,
        down_block: nn.Module,
        visual_tokenizer,
        pose_bundle_encoder,
        pose_conditioned_film,
        pose_feature_capture,
        injection_mode: str = "none",
        injection_scale_init: float = 0.0,
        detach_tokenizer_input: bool = True,
        detach_pose_encoder_input: bool = True,
    ) -> None:
        super().__init__()
        if injection_mode not in ("none", "residual"):
            raise ValueError(
                f"Unknown adapter_injection_mode={injection_mode!r}; expected 'none' or 'residual'"
            )
        object.__setattr__(self, "visual_tokenizer", visual_tokenizer)
        object.__setattr__(self, "pose_bundle_encoder", pose_bundle_encoder)
        object.__setattr__(self, "pose_conditioned_film", pose_conditioned_film)
        object.__setattr__(self, "pose_feature_capture", pose_feature_capture)

        self.injection_mode = injection_mode
        self.detach_tokenizer_input = detach_tokenizer_input
        self.detach_pose_encoder_input = detach_pose_encoder_input
        self.injection_scale = nn.Parameter(torch.tensor(float(injection_scale_init)))

        self.batch_size: Optional[int] = None
        self.num_frames: Optional[int] = None

        self.tokenizer_output: Optional[Dict[str, torch.Tensor]] = None
        self.pose_output: Optional[Dict[str, torch.Tensor]] = None
        self.film_output: Optional[Dict[str, torch.Tensor]] = None
        self.feature_for_unet: Optional[torch.Tensor] = None
        self.original_feature_5d: Optional[torch.Tensor] = None
        self._shapes_logged = False

        # Auto-detect (batch_size, num_frames) from the UNet's own `sample`
        # argument (always the first positional/kwarg arg, shape
        # `(batch, num_frames, channel, height, width)` -- verified against
        # UNetSpatioTemporalConditionModel.forward()'s real signature) via a
        # forward PRE-hook on the top-level UNet, fired right before every
        # unet(...) call regardless of caller. This matters because
        # train_vq.py always calls the UNet with a fixed (B, T), but
        # MimicMotionPipeline (mimicmotion/pipelines/pipeline_mimicmotion.py)
        # calls self.unet(...) many times per generation with DIFFERENT
        # batch/frame counts each time (batch_size=1 per CFG branch, and
        # num_frames=len(idx) which varies per tile, not the fixed
        # num_frames used during training) -- so a single manually-set value
        # would silently go stale. set_batch_context() below still exists for
        # explicit/manual use but is redundant once this pre-hook is
        # registered (the pre-hook always overwrites it with the live value
        # right before the down_blocks[1] hook fires in the same forward call).
        self._unet_pre_handle = unet.register_forward_pre_hook(self._unet_pre_hook, with_kwargs=True)
        self._handle = down_block.register_forward_hook(self._hook)

        if is_rank0():
            print(
                "[ContinuousAdapterInjector] "
                f"injection_mode={injection_mode}, injection_scale_init={injection_scale_init}, "
                f"detach_tokenizer_input={detach_tokenizer_input}, "
                f"detach_pose_encoder_input={detach_pose_encoder_input}, params=1 (injection_scale)"
            )

    def _unet_pre_hook(self, module: nn.Module, args, kwargs) -> None:
        sample = args[0] if len(args) > 0 else kwargs["sample"]
        if sample.dim() != 5:
            raise RuntimeError(
                "ContinuousAdapterInjector: expected the UNet's `sample` argument to be 5D "
                f"(batch, num_frames, channel, height, width), got shape={tuple(sample.shape)}"
            )
        self.batch_size, self.num_frames = sample.shape[0], sample.shape[1]

    def set_batch_context(self, batch_size: int, num_frames: int) -> None:
        """Optional manual override -- normally unnecessary since the
        forward-pre-hook on `unet` (registered in __init__) already captures
        (batch_size, num_frames) automatically from the real `sample` tensor
        on every unet(...) call, including inside inference pipelines with
        tiling/CFG where these values change per call."""
        self.batch_size = batch_size
        self.num_frames = num_frames

    def remove(self) -> None:
        self._handle.remove()
        self._unet_pre_handle.remove()

    def _hook(self, module: nn.Module, inputs, output: Tuple[torch.Tensor, tuple]):
        if self.batch_size is None or self.num_frames is None:
            raise RuntimeError(
                "ContinuousAdapterInjector: set_batch_context(batch_size, num_frames) "
                "must be called before the UNet forward pass that triggers this hook."
            )
        sample, res_samples = output
        if len(res_samples) == 0 or res_samples[-1] is not sample:
            raise RuntimeError(
                "ContinuousAdapterInjector: expected res_samples[-1] to be the exact same "
                "tensor object as the returned `sample` (true for this project's "
                "down_blocks[1], which has a downsampler -- verified against the installed "
                "diffusers CrossAttnDownBlockSpatioTemporal.forward() source, see module "
                "docstring). If the base UNet architecture changed, this assumption must be "
                "re-verified before injecting."
            )

        b, t = self.batch_size, self.num_frames
        feature_5d = restore_video_dimension(sample, batch_size=b, num_frames=t)
        self.original_feature_5d = feature_5d

        tokenizer_input = feature_5d.detach() if self.detach_tokenizer_input else feature_5d
        patches = self.visual_tokenizer.patchify(tokenizer_input)
        visual_tokens = self.visual_tokenizer.encode(patches)

        raw_pose_feature = self.pose_feature_capture.output
        if raw_pose_feature is None:
            raise RuntimeError(
                "ContinuousAdapterInjector: pose_feature_capture.output is None -- "
                "encode_pose_latents() must run BEFORE comps.unet(...) so the pose feature "
                "is already captured by the time this down_blocks[1] hook fires."
            )
        pose_input = raw_pose_feature.detach() if self.detach_pose_encoder_input else raw_pose_feature
        target_hw = (tokenizer_input.shape[-2], tokenizer_input.shape[-1])
        pose_output = self.pose_bundle_encoder(
            pose_input, target_spatial_size=target_hw,
            batch_size=b, num_frames=t, visual_tokens=visual_tokens,
        )
        self.pose_output = pose_output

        film_output = self.pose_conditioned_film(
            visual_tokens=visual_tokens,
            global_pose_embedding=pose_output["global_pose_embedding"],
            local_pose_occupancy=pose_output["local_pose_occupancy"],
        )
        self.film_output = film_output
        tokens_for_decode = film_output["modulated_tokens"]

        decoded_patches = self.visual_tokenizer.decode(tokens_for_decode)
        reconstructed = self.visual_tokenizer.unpatchify(decoded_patches, original_shape=tokenizer_input.shape)
        x32 = tokenizer_input.float()
        rec32 = reconstructed.float()
        reconstruction_loss = F.smooth_l1_loss(rec32, x32)
        relative_l2_error = torch.norm(rec32 - x32) / (torch.norm(x32) + 1e-8)
        cosine_similarity = F.cosine_similarity(rec32, x32, dim=2).mean()
        self.tokenizer_output = {
            "patches": patches,
            "tokens": visual_tokens,
            "decoded_patches": decoded_patches,
            "reconstructed": reconstructed,
            "reconstruction_loss": reconstruction_loss,
            "relative_l2_error": relative_l2_error,
            "cosine_similarity": cosine_similarity,
        }

        # Residual injection: at injection_scale=0 this is an exact identity
        # (feature_for_unet == feature_5d), regardless of injection_mode.
        if self.injection_mode == "residual":
            feature_delta = reconstructed - feature_5d
            feature_for_unet_5d = feature_5d + self.injection_scale * feature_delta
        else:
            feature_for_unet_5d = feature_5d
        self.feature_for_unet = feature_for_unet_5d
        feature_for_unet_4d = flatten_video_dimension(feature_for_unet_5d)

        if not self._shapes_logged and is_rank0():
            print(
                "[ContinuousAdapterInjector shapes] "
                f"original_selected_feature={tuple(sample.shape)} "
                f"adapter_visual_feature_5d={tuple(feature_5d.shape)} "
                f"reconstructed_feature={tuple(reconstructed.shape)} "
                f"feature_for_unet={tuple(feature_for_unet_5d.shape)} "
                f"feature_for_unet_restored_for_unet={tuple(feature_for_unet_4d.shape)} "
                f"injection_mode={self.injection_mode}"
            )
            self._shapes_logged = True

        if self.injection_mode == "none":
            return None

        new_res_samples = res_samples[:-1] + (feature_for_unet_4d,)
        return feature_for_unet_4d, new_res_samples
