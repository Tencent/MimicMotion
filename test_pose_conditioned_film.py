"""
test_pose_conditioned_film.py

pose_conditioned_film.py 단독 실행 가능한 shape/정확성/gradient 테스트.
프로젝트의 나머지 코드(dataset.py, utils.py, mimicmotion/*)에 의존하지 않는다.

실행:
    python test_pose_conditioned_film.py
"""

from __future__ import annotations

import torch

from pose_conditioned_film import PoseConditionedFiLM, pose_film_shuffle_diagnostic

B, NT, NH, NW, DV, DP = 2, 4, 8, 6, 64, 32


def _make_inputs():
    visual_tokens = torch.randn(B, NT, NH, NW, DV)
    global_pose_embedding = torch.randn(B, NT, DP)
    local_pose_occupancy = torch.rand(B, NT, NH, NW, 1)
    return visual_tokens, global_pose_embedding, local_pose_occupancy


def test_shapes_and_zero_init_identity() -> None:
    """기본(zero-zero) 초기화: shape이 스펙과 정확히 일치하고, modulated_tokens는
    visual_tokens와 정확히 동일해야 한다 (film_scale=0 초기화)."""
    visual_tokens, global_pose_embedding, local_pose_occupancy = _make_inputs()
    film = PoseConditionedFiLM(
        visual_embed_dim=DV, pose_embed_dim=DP, hidden_dim=128, film_scale_init=0.0,
    )

    output = film(visual_tokens, global_pose_embedding, local_pose_occupancy)

    assert output["modulated_tokens"].shape == visual_tokens.shape, output["modulated_tokens"].shape
    assert output["gamma"].shape == (B, NT, DV), output["gamma"].shape
    assert output["beta"].shape == (B, NT, DV), output["beta"].shape
    assert torch.allclose(output["modulated_tokens"], visual_tokens, atol=1e-6), \
        "modulated_tokens must equal visual_tokens exactly at film_scale=0 init"

    print("[OK] shapes match spec, modulated_tokens == visual_tokens at zero-init (atol=1e-6)")


def test_backward_runs_and_grad_tensors_exist() -> None:
    """기본(zero-zero) 초기화에서도 backward()는 정상 실행되고 film_mlp/film_scale의
    .grad 텐서는 존재해야 한다 (값 자체는 이중 zero-init 때문에 0일 수 있음 --
    이는 pose_conditioned_film.py 모듈 docstring에 기록된 예상된 동작이다)."""
    visual_tokens, global_pose_embedding, local_pose_occupancy = _make_inputs()
    film = PoseConditionedFiLM(
        visual_embed_dim=DV, pose_embed_dim=DP, hidden_dim=128, film_scale_init=0.0,
    )
    output = film(visual_tokens, global_pose_embedding, local_pose_occupancy)
    loss = output["modulated_tokens"].mean()
    loss.backward()

    assert film.film_scale.grad is not None, "film_scale did not receive a grad tensor at all"
    for name, p in film.film_mlp.named_parameters():
        assert p.grad is not None, f"film_mlp.{name} did not receive a grad tensor at all"
        assert torch.isfinite(p.grad).all(), f"film_mlp.{name} grad has NaN/Inf"

    print("[OK] backward() runs cleanly, film_scale/film_mlp all have grad tensors (finite)")


def test_backward_grad_is_nonzero_with_nonzero_film_scale_init() -> None:
    """섹션10 대안(2): film_scale_init을 0이 아닌 값(1e-3)으로 주면, 마지막
    Linear가 zero-init이어도(gamma/beta의 '값'은 여전히 0) film_mlp 마지막
    Linear의 '가중치에 대한 gradient'는 0이 아니게 된다 (film_scale이 곱해지는
    체인룰 계수가 이제 0이 아니기 때문). 동시에 modulated_tokens는 여전히
    visual_tokens와 동일해야 한다(gamma=beta=0이므로 film_delta=0)."""
    visual_tokens, global_pose_embedding, local_pose_occupancy = _make_inputs()
    film = PoseConditionedFiLM(
        visual_embed_dim=DV, pose_embed_dim=DP, hidden_dim=128,
        film_scale_init=1e-3, last_linear_init="zero",
    )
    output = film(visual_tokens, global_pose_embedding, local_pose_occupancy)
    assert torch.allclose(output["modulated_tokens"], visual_tokens, atol=1e-6), \
        "gamma/beta are still exactly 0 at this init, so modulated_tokens must still equal visual_tokens"

    loss = output["modulated_tokens"].pow(2).mean()
    loss.backward()

    last_linear = film.film_mlp[-1]
    assert last_linear.weight.grad is not None
    nonzero = last_linear.weight.grad.abs().sum().item() > 0
    assert nonzero, (
        "expected non-zero gradient into film_mlp's last Linear when film_scale_init != 0, "
        "even though the last Linear itself is zero-initialized"
    )
    print("[OK] film_scale_init=1e-3 unblocks non-zero gradient into the zero-initialized "
          "last Linear layer, while modulated_tokens stays identical to visual_tokens at init")


def test_shape_validation_errors() -> None:
    visual_tokens, global_pose_embedding, local_pose_occupancy = _make_inputs()
    film = PoseConditionedFiLM(visual_embed_dim=DV, pose_embed_dim=DP, hidden_dim=128)

    try:
        film(visual_tokens, global_pose_embedding[:, :2], local_pose_occupancy)
        raise AssertionError("should have raised ValueError for Nt mismatch")
    except ValueError as e:
        assert "Nt" in str(e) or "B,Nt" in str(e)

    try:
        film(visual_tokens, global_pose_embedding, local_pose_occupancy[:, :, :4])
        raise AssertionError("should have raised ValueError for Nh mismatch")
    except ValueError as e:
        assert "Nh" in str(e) or "Nw" in str(e)

    print("[OK] shape mismatches raise clear ValueError")


def test_pose_film_shuffle_diagnostic_detects_real_variation() -> None:
    """서로 다른 temporal bundle마다 뚜렷하게 다른 pose embedding/occupancy를
    주면(진짜 pose 변화 근사), film_scale!=0이고 gamma/beta가 0이 아닐 때
    shuffle로 modulated_tokens가 실제로 달라져야 한다."""
    torch.manual_seed(0)
    visual_tokens = torch.randn(B, NT, NH, NW, DV)
    global_pose_embedding = torch.randn(B, NT, DP) * torch.arange(1, NT + 1).view(1, NT, 1).float()
    local_pose_occupancy = torch.rand(B, NT, NH, NW, 1)

    film = PoseConditionedFiLM(
        visual_embed_dim=DV, pose_embed_dim=DP, hidden_dim=128,
        film_scale_init=1.0, last_linear_init="small_normal",
    )
    result = pose_film_shuffle_diagnostic(film, visual_tokens, global_pose_embedding, local_pose_occupancy)
    assert torch.isfinite(torch.tensor(result["shuffle_modulated_token_l1_diff"]))
    assert result["shuffle_modulated_token_l1_diff"] > 1e-6, (
        "expected shuffling pose order to noticeably change modulated_tokens when "
        "film_scale=1.0 and gamma/beta are non-zero"
    )
    print("[OK] pose_film_shuffle_diagnostic detects real pose variation:", result)


if __name__ == "__main__":
    test_shapes_and_zero_init_identity()
    test_backward_runs_and_grad_tensors_exist()
    test_backward_grad_is_nonzero_with_nonzero_film_scale_init()
    test_shape_validation_errors()
    test_pose_film_shuffle_diagnostic_detects_real_variation()
    print("\nAll pose_conditioned_film tests passed.")
