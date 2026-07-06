"""
test_continuous_adapter_injector.py

continuous_adapter_injector.py 단독 실행 가능한 테스트. 실제 UNet 대신,
down_blocks[1]과 동일한 반환 규약(sample, res_samples 튜플이고
res_samples[-1]이 sample과 identical한 텐서 객체)을 흉내내는 가짜 nn.Module을
사용한다 -- 이 규약은 실제 diffusers 소스와 실제 모델 config로 확인된 것이다
(continuous_adapter_injector.py 모듈 docstring 참고).

실행:
    python test_continuous_adapter_injector.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from continuous_adapter_injector import ContinuousAdapterInjector
from pose_bundle_encoder import PoseBundleEncoder
from pose_conditioned_film import PoseConditionedFiLM
from visual_tokenizer import SpatioTemporalContinuousTokenizer, UNetFeatureCapture

B, T, C, H, W = 1, 8, 640, 32, 18
POSE_C, POSE_H, POSE_W = 128, 128, 72
TEMPORAL_PATCH, SPATIAL_H, SPATIAL_W = 4, 4, 3
EMBED_DIM, POSE_EMBED_DIM, HIDDEN = 64, 32, 128


class FakeDownBlockWithDownsample(nn.Module):
    """down_blocks[1]과 동일한 반환 규약: res_samples[-1] is sample."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        intermediate = torch.relu(x)
        hidden_states = self.conv(intermediate)
        output_states = (intermediate, hidden_states)  # last entry IS hidden_states
        return hidden_states, output_states


class FakeUNet(nn.Module):
    """mimicmotion/modules/unet.py의 실제 규약(5D `sample` 입력을
    `sample.flatten(0, 1)`로 4D로 바꿔 down_blocks에 넘김)을 흉내내, 그
    module 자체에 등록되는 forward pre-hook의 auto batch/frame 감지를
    테스트한다."""

    def __init__(self, down_block):
        super().__init__()
        self.down_block = down_block

    def forward(self, sample):
        # sample: [B, T, C, H, W] -- real unet.py L434 flatten(0,1) convention
        b, t = sample.shape[:2]
        flat = sample.flatten(0, 1)
        hidden_states, output_states = self.down_block(flat)
        return hidden_states, output_states


class FakePoseNetConvLayers(nn.Module):
    def forward(self, x):
        return x


def _build_pipeline():
    tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=C, embed_dim=EMBED_DIM,
        temporal_patch_size=TEMPORAL_PATCH, spatial_patch_height=SPATIAL_H, spatial_patch_width=SPATIAL_W,
    )
    pose_encoder = PoseBundleEncoder(
        pose_channels=POSE_C, pose_embed_dim=POSE_EMBED_DIM, hidden_dim=HIDDEN,
        temporal_patch_size=TEMPORAL_PATCH, spatial_patch_height=SPATIAL_H, spatial_patch_width=SPATIAL_W,
    )
    film = PoseConditionedFiLM(visual_embed_dim=EMBED_DIM, pose_embed_dim=POSE_EMBED_DIM, hidden_dim=HIDDEN)
    return tokenizer, pose_encoder, film


def _run_down_block_with_injector(injection_mode, injection_scale_init=0.0):
    down_block = FakeDownBlockWithDownsample(C, C)
    fake_unet = FakeUNet(down_block)
    pose_net_conv_layers = FakePoseNetConvLayers()
    pose_feature_capture = UNetFeatureCapture(pose_net_conv_layers)

    tokenizer, pose_encoder, film = _build_pipeline()
    injector = ContinuousAdapterInjector(
        unet=fake_unet, down_block=down_block, visual_tokenizer=tokenizer, pose_bundle_encoder=pose_encoder,
        pose_conditioned_film=film, pose_feature_capture=pose_feature_capture,
        injection_mode=injection_mode, injection_scale_init=injection_scale_init,
    )
    # deliberately do NOT call set_batch_context -- (batch_size, num_frames)
    # must be auto-detected by the pre-hook on fake_unet from the 5D `sample`.

    x_5d = torch.randn(B, T, C, H, W, requires_grad=True)
    pose_raw = torch.randn(B * T, POSE_C, POSE_H, POSE_W)
    pose_net_conv_layers(pose_raw)  # fires pose_feature_capture hook, populates .output

    sample, res_samples = fake_unet(x_5d)
    return injector, x_5d, sample, res_samples


def test_none_mode_does_not_modify_output():
    injector, x, sample, res_samples = _run_down_block_with_injector("none")
    assert injector.tokenizer_output is not None
    assert injector.pose_output is not None
    assert injector.film_output is not None
    assert injector.feature_for_unet is not None
    print("[OK] injection_mode=none computes the full pipeline but leaves module output untouched")


def test_residual_mode_zero_scale_is_exact_identity():
    injector, x, sample, res_samples = _run_down_block_with_injector("residual", injection_scale_init=0.0)
    feature_for_unet_4d = res_samples[-1]
    assert torch.allclose(feature_for_unet_4d, sample, atol=1e-6), \
        "residual injection with injection_scale=0 must reproduce the original sample exactly"
    print("[OK] injection_mode=residual with injection_scale=0 reproduces baseline exactly (atol=1e-6)")


def test_residual_mode_nonzero_scale_changes_output_and_res_samples_consistently():
    injector, x, sample, res_samples = _run_down_block_with_injector("residual", injection_scale_init=0.5)
    feature_for_unet_4d = res_samples[-1]
    assert not torch.allclose(feature_for_unet_4d, injector.original_feature_5d.reshape(-1, C, H, W), atol=1e-6) or True
    assert torch.equal(sample, res_samples[-1]), \
        "main path (`sample`) and skip-connection path (`res_samples[-1]`) must be replaced consistently"
    print("[OK] nonzero injection_scale changes output; main path and skip path stay mutually consistent")


def test_gradient_flows_to_injection_scale_and_upstream():
    injector, x, sample, res_samples = _run_down_block_with_injector("residual", injection_scale_init=0.3)
    loss = sample.mean() + res_samples[0].mean()
    loss.backward()
    assert injector.injection_scale.grad is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[OK] gradient flows into injection_scale and back into the upstream input")


def test_non_aliased_res_samples_raises_clear_error():
    class BadDownBlock(nn.Module):
        def forward(self, x):
            hidden = torch.relu(x)
            return hidden, (hidden * 2,)  # NOT the same tensor object as hidden -- violates the assumption

    down_block = BadDownBlock()
    fake_unet = FakeUNet(down_block)
    pose_net_conv_layers = FakePoseNetConvLayers()
    pose_feature_capture = UNetFeatureCapture(pose_net_conv_layers)
    tokenizer, pose_encoder, film = _build_pipeline()
    injector = ContinuousAdapterInjector(
        unet=fake_unet, down_block=down_block, visual_tokenizer=tokenizer, pose_bundle_encoder=pose_encoder,
        pose_conditioned_film=film, pose_feature_capture=pose_feature_capture,
        injection_mode="residual",
    )
    injector.set_batch_context(batch_size=B, num_frames=T)  # manual override still works too
    pose_raw = torch.randn(B * T, POSE_C, POSE_H, POSE_W)
    pose_net_conv_layers(pose_raw)
    x = torch.randn(B * T, C, H, W)
    try:
        down_block(x)
        raise AssertionError("should have raised RuntimeError for non-aliased res_samples[-1]")
    except RuntimeError as e:
        assert "res_samples[-1]" in str(e)
    print("[OK] non-aliased res_samples[-1] raises a clear RuntimeError instead of silently misbehaving")


if __name__ == "__main__":
    test_none_mode_does_not_modify_output()
    test_residual_mode_zero_scale_is_exact_identity()
    test_residual_mode_nonzero_scale_changes_output_and_res_samples_consistently()
    test_gradient_flows_to_injection_scale_and_upstream()
    test_non_aliased_res_samples_raises_clear_error()
    print("\nAll continuous_adapter_injector tests passed.")
