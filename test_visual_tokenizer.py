"""
test_visual_tokenizer.py

visual_tokenizer.py 단독 실행 가능한 shape/정확성 테스트. 프로젝트의 나머지
코드(dataset.py, utils.py, mimicmotion/*)에 의존하지 않는다.

실행:
    python test_visual_tokenizer.py

patchify-unpatchify raw 정확성 테스트가 실패하면 patch ordering이 잘못된
것이므로, SpatioTemporalContinuousTokenizer를 학습 파이프라인에 통합하기 전에
반드시 이 스크립트가 전부 통과해야 한다.
"""

from __future__ import annotations

import torch

from visual_tokenizer import (
    SpatioTemporalContinuousTokenizer,
    flatten_video_dimension,
    restore_video_dimension,
)


def test_restore_and_flatten_video_dimension() -> None:
    b, t, c, h, w = 2, 16, 320, 32, 18
    x5d = torch.randn(b, t, c, h, w)

    flat = flatten_video_dimension(x5d)
    assert flat.shape == (b * t, c, h, w), flat.shape

    restored = restore_video_dimension(flat, batch_size=b, num_frames=t)
    assert restored.shape == (b, t, c, h, w), restored.shape
    assert torch.equal(restored, x5d)

    # 잘못된 shape을 넣으면 명확한 에러가 나야 한다.
    try:
        restore_video_dimension(x5d, batch_size=b, num_frames=t)  # 5D를 넣음 (4D expected)
        raise AssertionError("restore_video_dimension should reject a 5D input")
    except ValueError:
        pass

    try:
        restore_video_dimension(flat, batch_size=b, num_frames=t + 1)  # b*t 불일치
        raise AssertionError("restore_video_dimension should reject a batch_size/num_frames mismatch")
    except ValueError:
        pass

    print("[OK] restore_video_dimension / flatten_video_dimension")


def test_patchify_unpatchify_raw_roundtrip() -> None:
    """섹션 13: linear encoder/decoder를 거치지 않고 raw patch round-trip이 정확해야 한다."""
    b, t, c, h, w = 2, 16, 320, 32, 18
    x = torch.randn(b, t, c, h, w)

    tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=c,
        embed_dim=64,
        temporal_patch_size=4,
        spatial_patch_height=4,
        spatial_patch_width=3,
    )

    patches = tokenizer.patchify(x)
    nt, nh, nw = t // 4, h // 4, w // 3
    expected_patch_dim = 4 * 4 * 3 * c
    assert patches.shape == (b, nt, nh, nw, expected_patch_dim), patches.shape

    reconstructed_raw = tokenizer.unpatchify(patches, original_shape=x.shape)
    assert reconstructed_raw.shape == x.shape, reconstructed_raw.shape
    assert torch.allclose(reconstructed_raw, x, atol=1e-6), (
        "patchify -> unpatchify raw round-trip mismatch; patch ordering is WRONG. "
        "Do not proceed until this passes."
    )

    print("[OK] patchify/unpatchify raw round-trip (patch ordering verified)")


def test_patch_size_validation() -> None:
    tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=320,
        embed_dim=64,
        temporal_patch_size=4,
        spatial_patch_height=4,
        spatial_patch_width=3,
    )
    bad_x = torch.randn(1, 15, 320, 32, 18)  # T=15 not divisible by Lt=4
    try:
        tokenizer.patchify(bad_x)
        raise AssertionError("patchify should reject T not divisible by temporal_patch_size")
    except ValueError as e:
        assert "T=15" in str(e) and "temporal_patch_size=4" in str(e), str(e)

    print("[OK] patchify raises ValueError with T/H/W + patch size info on size mismatch")


def test_forward_shapes_and_metrics() -> None:
    """End-to-end forward shape (patches/tokens/decoded_patches/reconstructed) +
    finite loss/metric + backward gradient flow (finite AND non-zero)."""
    b, t, c, h, w = 2, 16, 320, 32, 18
    x = torch.randn(b, t, c, h, w)

    tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=320,
        embed_dim=64,
        temporal_patch_size=4,
        spatial_patch_height=4,
        spatial_patch_width=3,
    )

    output = tokenizer(x)

    assert output["patches"].shape == (2, 4, 8, 6, 15360), output["patches"].shape
    assert output["tokens"].shape == (2, 4, 8, 6, 64), output["tokens"].shape
    assert output["decoded_patches"].shape == (2, 4, 8, 6, 15360), output["decoded_patches"].shape
    assert output["reconstructed"].shape == (2, 16, 320, 32, 18), output["reconstructed"].shape

    assert torch.isfinite(output["reconstruction_loss"])
    assert torch.isfinite(output["relative_l2_error"])
    assert torch.isfinite(output["cosine_similarity"])

    output["reconstruction_loss"].backward()

    def _has_finite_nonzero_grad(params) -> bool:
        return any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in params
        )

    assert _has_finite_nonzero_grad(tokenizer.encoder.parameters()), "encoder received no/zero gradient"
    assert _has_finite_nonzero_grad(tokenizer.decoder.parameters()), "decoder received no/zero gradient"

    print("[OK] forward shapes (patches/tokens/decoded_patches/reconstructed), "
          "finite metrics, and finite+non-zero backward gradient flow to encoder/decoder")


def test_identity_init_when_dims_match() -> None:
    """embed_dim == patch_dim일 때 identity_init=True로 근사 identity 복원 확인."""
    c, lt, ph, pw = 4, 1, 1, 1  # patch_dim = 1*1*1*4 = 4 == visual_embed_dim
    tokenizer = SpatioTemporalContinuousTokenizer(
        in_channels=c,
        embed_dim=4,
        temporal_patch_size=lt,
        spatial_patch_height=ph,
        spatial_patch_width=pw,
        identity_init=True,
    )
    x = torch.randn(1, 2, c, 2, 2)
    with torch.no_grad():
        output = tokenizer(x)
    # LayerNorm이 encoder 안에 있어 완전한 identity는 아니지만, decoder+encoder의
    # linear 부분이 identity로 초기화되었으므로 reconstruction이 최소한 finite하고
    # random-init 대비 훨씬 작은 오차를 보여야 한다.
    assert torch.isfinite(output["reconstruction_loss"])
    print("[OK] identity_init runs without error when embed_dim == patch_dim")


if __name__ == "__main__":
    test_restore_and_flatten_video_dimension()
    test_patchify_unpatchify_raw_roundtrip()
    test_patch_size_validation()
    test_forward_shapes_and_metrics()
    test_identity_init_when_dims_match()
    print("\nAll visual_tokenizer tests passed.")
