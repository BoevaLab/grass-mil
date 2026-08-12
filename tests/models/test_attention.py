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
