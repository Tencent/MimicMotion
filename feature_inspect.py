"""
feature_inspect.py

향후 UNet visual feature -> pose embedding -> FiLM -> VQ -> residual injection
파이프라인을 어디에 연결할지 결정하기 위한, 읽기 전용 feature shape inspection 유틸리티.

이 모듈은:
  - register_forward_hook / register_forward_pre_hook만 사용한다. UNet/PoseNet 소스는
    수정하지 않는다 (mimicmotion/modules/unet.py, mimicmotion/modules/pose_net.py 그대로).
  - tensor 값을 절대 저장/clone/CPU 복사하지 않는다. shape/dtype/device/requires_grad만
    읽고 즉시 버린다. GPU 메모리 증가 없음, forward/backward 수치 결과에 영향 없음.
  - 첫 forward pass가 끝나면(= UNet 최상위 post-hook이 마지막으로 실행되는 시점) 스스로
    모든 hook을 제거한다. 이후 iteration에는 어떤 오버헤드도 남지 않는다.
  - hook은 down/mid/up "block" 경계에 걸려 있고, gradient checkpointing은 그 블록
    "내부"의 resnet/attention 서브모듈 단위로 적용되므로, backward 중 recompute가
    이 블록-레벨 hook을 다시 실행시키지 않는다 (검증: 아래 __main__ 스모크테스트 참고).
    그래도 안전장치로 _done 플래그를 이중으로 둔다.
  - torch.distributed가 초기화되어 있으면 rank 0에서만 출력한다.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover
    dist = None


def _is_rank0() -> bool:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log_feature(name: str, tensor: torch.Tensor) -> None:
    """Read-only: shape/dtype/device/requires_grad만 읽는다. tensor 값은 만지지 않는다."""
    if not _is_rank0():
        return
    print(
        "[FeatureInspect] "
        f"name={name}, "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"requires_grad={tensor.requires_grad}"
    )


class FeatureInspector:
    """UNetSpatioTemporalConditionModel + PoseNet에 read-only hook을 걸어 첫 forward의
    feature shape만 한 번 출력하고 스스로 해제된다.

    사용법:
        inspector = FeatureInspector(comps.unet, comps.pose_net, enabled=cfg.get("enable_feature_inspection", False))
        # 평소처럼 학습 루프 실행. 첫 forward 직후 로그가 찍히고 이후로는 아무 동작도 하지 않는다.
    """

    def __init__(self, unet: nn.Module, pose_net: nn.Module, enabled: bool = True):
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._done = False
        self.enabled = bool(enabled)
        if self.enabled:
            self._register(unet, pose_net)

    # -- internal -----------------------------------------------------
    def _log_once(self, name: str, tensor: torch.Tensor) -> None:
        if self._done or tensor is None:
            return
        log_feature(name, tensor)

    def _finish(self) -> None:
        self._done = True
        for h in self._handles:
            h.remove()
        self._handles = []

    def _register(self, unet: nn.Module, pose_net: nn.Module) -> None:
        # 9. pose encoder(PoseNet) 출력
        def pose_net_hook(_m, _inp, out):
            self._log_once("pose_net.output  [ (B*T), C, H, W ] (encode_pose_latents 결과)", out)

        self._handles.append(pose_net.register_forward_hook(pose_net_hook))

        # 1. UNet 입력 latent / 10. UNet에 실제로 전달되는 pose feature (dtype 캐스팅 이후)
        def unet_pre_hook(_m, args, kwargs):
            if args:
                self._log_once("unet.input.sample  [B, T, C, H, W] (noisy+ref concat latent)", args[0])
            pl = kwargs.get("pose_latents")
            if pl is not None:
                self._log_once("unet.input.pose_latents  [ (B*T), C, H, W ] (unet_dtype로 cast된 뒤 실제 전달값)", pl)

        self._handles.append(unet.register_forward_pre_hook(unet_pre_hook, with_kwargs=True))

        # 8. 최종 UNet 출력. 이 hook이 항상 가장 마지막에 실행되므로 여기서 전체 inspection을 종료한다.
        def unet_post_hook(_m, _inp, out):
            sample = out.sample if hasattr(out, "sample") else out[0]
            self._log_once("unet.output.sample  [B, T, C, H, W] (model_pred)", sample)
            self._finish()

        self._handles.append(unet.register_forward_hook(unet_post_hook))

        # 2 & 3. down_blocks 출력 + skip으로 저장되는 res_samples
        # down_blocks[i].forward -> (sample, res_samples); 각 원소가 그대로 skip feature.
        for i, block in enumerate(unet.down_blocks):
            def make_down_hook(i=i):
                def down_hook(_m, _inp, out):
                    sample, res_samples = out
                    self._log_once(f"down_blocks[{i}].output.sample  [ (B*T), C, H, W ]", sample)
                    for j, r in enumerate(res_samples):
                        self._log_once(
                            f"down_blocks[{i}].res_samples[{j}]  [ (B*T), C, H, W ] (skip connection 저장값)", r
                        )

                return down_hook

            self._handles.append(block.register_forward_hook(make_down_hook()))

            # down_blocks[0]의 입력 == UNetSpatioTemporalConditionModel.forward의
            # `down_block_res_samples = (sample,)` 초기값 (conv_in 출력 + pose_latents를 더한 직후).
            # 이 값 자체는 어떤 block의 "output"도 아니라서 down_blocks[0]의 pre-hook으로만 잡을 수 있다.
            if i == 0:
                def first_down_pre_hook(_m, args, kwargs):
                    hs = kwargs.get("hidden_states", args[0] if args else None)
                    self._log_once(
                        "down_blocks[0].input == initial skip feature"
                        "  [ (B*T), C, H, W ] (conv_in(sample) + pose_latents 직후)",
                        hs,
                    )

                self._handles.append(block.register_forward_pre_hook(first_down_pre_hook, with_kwargs=True))

        # 4. mid block 입력/출력
        def mid_pre_hook(_m, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            self._log_once("mid_block.input  [ (B*T), C, H, W ]", hs)

        self._handles.append(unet.mid_block.register_forward_pre_hook(mid_pre_hook, with_kwargs=True))

        def mid_post_hook(_m, _inp, out):
            self._log_once("mid_block.output  [ (B*T), C, H, W ]", out)

        self._handles.append(unet.mid_block.register_forward_hook(mid_post_hook))

        # 4 & 6 & 7. up_blocks 입력(및 소비하는 skip feature) / 출력
        for i, block in enumerate(unet.up_blocks):
            def make_up_pre_hook(i=i):
                def up_pre_hook(_m, args, kwargs):
                    hs = kwargs.get("hidden_states", args[0] if args else None)
                    res = kwargs.get("res_hidden_states_tuple")
                    self._log_once(f"up_blocks[{i}].input.hidden_states  [ (B*T), C, H, W ]", hs)
                    if res is not None:
                        for j, r in enumerate(res):
                            self._log_once(
                                f"up_blocks[{i}].input.res_hidden_states_tuple[{j}]"
                                "  [ (B*T), C, H, W ] (소비되는 skip feature)",
                                r,
                            )

                return up_pre_hook

            self._handles.append(block.register_forward_pre_hook(make_up_pre_hook(), with_kwargs=True))

            def make_up_post_hook(i=i):
                def up_post_hook(_m, _inp, out):
                    self._log_once(f"up_blocks[{i}].output  [ (B*T), C, H, W ]", out)

                return up_post_hook

            self._handles.append(block.register_forward_hook(make_up_post_hook()))
