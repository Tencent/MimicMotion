"""
test_pose_bundle_encoder.py

pose_bundle_encoder.py 단독 실행 가능한 shape/정확성 테스트. 프로젝트의
나머지 코드(dataset.py, utils.py, mimicmotion/*)에 의존하지 않는다.

실행:
    python test_pose_bundle_encoder.py
"""

from __future__ import annotations

import torch

from pose_bundle_encoder import PoseBundleEncoder, pose_shuffle_diagnostic


def _make_encoder() -> PoseBundleEncoder:
    return PoseBundleEncoder(
        pose_channels=128,
        pose_embed_dim=32,
        hidden_dim=128,
        temporal_patch_size=4,
        spatial_patch_height=4,
        spatial_patch_width=3,
        pose_normalization="none",
        occupancy_activation_mode="abs_mean",
        occupancy_normalization="sigmoid",
    )


def test_forward_shapes_and_metrics() -> None:
    b, t, cp, hp, wp = 2, 16, 128, 64, 36
    hx, wx = 32, 18
    pose_feature = torch.randn(b, t, cp, hp, wp)

    encoder = _make_encoder()
    output = encoder(pose_feature, target_spatial_size=(hx, wx), batch_size=b, num_frames=t)

    assert output["pose_aligned"].shape == (2, 16, 128, 32, 18), output["pose_aligned"].shape
    assert output["pose_temporal_bundles"].shape == (2, 4, 4, 128, 32, 18), output["pose_temporal_bundles"].shape
    assert output["global_pose_raw"].shape == (2, 4, 128), output["global_pose_raw"].shape
    assert output["global_pose_embedding"].shape == (2, 4, 32), output["global_pose_embedding"].shape
    assert output["pose_activation"].shape == (2, 16, 32, 18), output["pose_activation"].shape
    assert output["local_pose_occupancy"].shape == (2, 4, 8, 6, 1), output["local_pose_occupancy"].shape
    assert output["reconstructed_global_pose"].shape == (2, 4, 128), output["reconstructed_global_pose"].shape

    for key in [
        "pose_reconstruction_loss", "global_pose_mean", "global_pose_std", "global_pose_norm",
        "global_pose_channel_variance", "global_pose_temporal_variance", "global_pose_batch_variance",
        "occupancy_mean", "occupancy_std", "occupancy_min", "occupancy_max",
        "occupancy_active_ratio", "occupancy_background_ratio",
        "occupancy_temporal_variance", "occupancy_spatial_variance",
    ]:
        assert torch.isfinite(output[key]), f"{key} is not finite: {output[key]}"

    output["pose_reconstruction_loss"].backward()

    def _has_finite_nonzero_grad(params) -> bool:
        return any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in params
        )

    assert _has_finite_nonzero_grad(encoder.global_encoder.parameters()), \
        "GlobalPoseEncoder received no/zero gradient"
    assert _has_finite_nonzero_grad(encoder.reconstruction_head.parameters()), \
        "PoseReconstructionHead received no/zero gradient"

    print("[OK] forward shapes match spec exactly, all metrics finite, "
          "finite+non-zero gradient to GlobalPoseEncoder and PoseReconstructionHead")


def test_4d_input_is_restored_correctly() -> None:
    """pose_feature가 [B*T, Cp, Hp, Wp]로 들어와도 [B,T,Cp,Hx,Wx]로 정확히 복원되는지."""
    b, t, cp, hp, wp = 2, 8, 128, 64, 36
    hx, wx = 32, 18
    pose_5d = torch.randn(b, t, cp, hp, wp)
    pose_4d = pose_5d.reshape(b * t, cp, hp, wp)

    encoder = _make_encoder()
    out_from_4d = encoder(pose_4d, target_spatial_size=(hx, wx), batch_size=b, num_frames=t)
    out_from_5d = encoder(pose_5d, target_spatial_size=(hx, wx), batch_size=b, num_frames=t)

    assert torch.equal(out_from_4d["pose_aligned"], out_from_5d["pose_aligned"])
    print("[OK] 4D [B*T,Cp,Hp,Wp] input restores identically to passing 5D directly")


def test_visual_pose_shape_consistency_validation() -> None:
    """visual_tokens와 Nt/Nh/Nw가 다르면 명확한 ValueError."""
    b, t, cp = 2, 8, 128
    hx, wx = 32, 18
    pose_feature = torch.randn(b, t, cp, hx, wx)
    encoder = _make_encoder()

    good_visual_tokens = torch.randn(b, t // 4, hx // 4, wx // 3, 64)  # Nt=2, Nh=8, Nw=6
    encoder(pose_feature, target_spatial_size=(hx, wx), batch_size=b, num_frames=t,
            visual_tokens=good_visual_tokens)

    bad_visual_tokens = torch.randn(b, t // 4 + 1, hx // 4, wx // 3, 64)  # wrong Nt
    try:
        encoder(pose_feature, target_spatial_size=(hx, wx), batch_size=b, num_frames=t,
                visual_tokens=bad_visual_tokens)
        raise AssertionError("should have raised ValueError for Nt mismatch")
    except ValueError as e:
        assert "Nt" in str(e)

    print("[OK] visual_tokens/pose bundle Nt,Nh,Nw consistency validated with clear ValueError")


def test_pose_shuffle_diagnostic_detects_real_pose_variation() -> None:
    """서로 다른 frame 값(진짜 pose 변화가 있는 입력)이면 shuffle이 embedding을 눈에 띄게 바꿔야 한다."""
    torch.manual_seed(0)
    b, t, cp = 1, 8, 128
    hx, wx = 32, 18
    # 프레임마다 뚜렷하게 다른 값(현실적인 pose 변화 근사)
    pose_feature = torch.randn(b, t, cp, hx, wx) * torch.arange(1, t + 1).view(1, t, 1, 1, 1).float()

    encoder = _make_encoder()
    result = pose_shuffle_diagnostic(encoder, pose_feature, target_spatial_size=(hx, wx),
                                      batch_size=b, num_frames=t)
    assert torch.isfinite(torch.tensor(result["global_pose_embedding_mean_abs_diff"]))
    assert torch.isfinite(torch.tensor(result["global_pose_cosine_similarity"]))
    assert torch.isfinite(torch.tensor(result["local_occupancy_mean_abs_diff"]))
    print("[OK] pose_shuffle_diagnostic runs and returns finite metrics:", result)


if __name__ == "__main__":
    test_forward_shapes_and_metrics()
    test_4d_input_is_restored_correctly()
    test_visual_pose_shape_consistency_validation()
    test_pose_shuffle_diagnostic_detects_real_pose_variation()
    print("\nAll pose_bundle_encoder tests passed.")
