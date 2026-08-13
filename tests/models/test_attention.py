import pytest
import torch


def test_gated_attention_shape():
    from grass_mil.models.components.attention import AttnNetGated

    module = AttnNetGated(input_dim=32, hidden_dim=16, dropout=False, n_classes=1)
    x = torch.randn(20, 32)
    logits, out_x = module(x)
    assert logits.shape == (20, 1)
    assert out_x.shape == x.shape


def test_gated_projected_attention_shape():
    from grass_mil.models.components.attention import AttnNetGatedProjected

    module = AttnNetGatedProjected(
        input_dim=32, projection_dim=16, hidden_dim=8, dropout=False, n_classes=1
    )
    x = torch.randn(20, 32)
    logits, out_x = module(x)
    assert logits.shape == (20, 1)
    assert out_x.shape == x.shape


@pytest.mark.parametrize("enabled", [True, False])
def test_attention_optional_factory(enabled):
    from grass_mil.models.components.factory import build_attention

    att = build_attention(enabled, attention_type="gated", input_dim=16, hidden_dim=8)
    if enabled:
        assert att is not None
    else:
        assert att is None


def test_attention_head_spaces_expose_the_pre_tanh_activation():
    """The interpretability suite clusters the pre-tanh attention space.

    Reaching into module internals from the analysis code is fragile, so the
    head exposes it. Pre-tanh matters because tanh saturation collapses the
    extremes that distinguish strongly-attended instances.
    """
    import torch

    from grass_mil.models.components.attention import AttnNetGated, AttnNetGatedProjected

    x = torch.randn(7, 12)

    gated = AttnNetGated(input_dim=12, hidden_dim=5, n_classes=2).eval()
    spaces = gated.head_spaces(x)
    assert set(spaces) == {"a_pre_tanh", "gate_product"}
    assert spaces["a_pre_tanh"].shape == (7, 5)
    # It really is pre-saturation: tanh of it reproduces the gate branch.
    torch.testing.assert_close(torch.tanh(spaces["a_pre_tanh"]), gated.attention_a(x))

    projected = AttnNetGatedProjected(
        input_dim=12, projection_dim=6, hidden_dim=4, n_classes=2
    ).eval()
    proj_spaces = projected.head_spaces(x)
    assert proj_spaces["a_pre_tanh"].shape == (7, 4)
    hidden = projected.projection(x)
    torch.testing.assert_close(
        torch.tanh(proj_spaces["a_pre_tanh"]), projected.attention_a(hidden)
    )
    # The pre-tanh space is unbounded, unlike the saturated gate output.
    assert proj_spaces["gate_product"].abs().max() <= 1.0
