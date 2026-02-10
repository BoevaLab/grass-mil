import pytest
import torch
from torch import nn


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return self.lin(x)


def test_bgrl_forward_and_update():
    from src.models.components.ssl import BGRL, MLPPredictor

    encoder = _TinyEncoder()
    predictor = MLPPredictor(input_size=8, output_size=8, hidden_size=16)
    model = BGRL(encoder=encoder, predictor=predictor)

    x1 = torch.randn(10, 8)
    x2 = torch.randn(10, 8)
    q, y = model(x1, x2)
    assert q.shape == (10, 8)
    assert y.shape == (10, 8)

    before = [p.clone() for p in model.target_encoder.parameters()]
    model.update_target_network(momentum=0.9)
    after = list(model.target_encoder.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


@pytest.mark.parametrize("enabled", [True, False])
def test_ssl_optional_factory(enabled):
    from src.models.components.backbones import EncoderConfig, GNNEncoder
    from src.models.components.factory import build_ssl

    encoder = GNNEncoder(
        EncoderConfig(input_dim=4, hidden_dim=8, out_dim=8, num_layers=1)
    )
    ssl_model = build_ssl(
        enabled,
        encoder,
        {
            "method": "bgrl",
            "predictor": {"hidden_size": 16},
        },
    )
    if enabled:
        assert ssl_model is not None
    else:
        assert ssl_model is None


def test_ssl_factory_requires_hidden_size():
    from src.models.components.backbones import EncoderConfig, GNNEncoder
    from src.models.components.factory import build_ssl

    encoder = GNNEncoder(
        EncoderConfig(input_dim=4, hidden_dim=8, out_dim=8, num_layers=1)
    )
    with pytest.raises(ValueError, match="ssl.predictor.hidden_size is required"):
        build_ssl(True, encoder, {"method": "bgrl", "predictor": {}})
