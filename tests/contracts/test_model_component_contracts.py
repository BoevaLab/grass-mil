from pathlib import Path


def test_contract_markdown_exists_and_has_required_sections():
    path = Path("docs/model_component_contracts.md")
    assert path.exists()
    text = path.read_text()
    assert "Graph Unit Contract" in text
    assert "Loss Contracts" in text
    assert "Sampler Contracts" in text


def test_cursor_rule_exists_and_mentions_data_layer_agnosticity():
    path = Path(".cursor/rules/model_component_contracts.mdc")
    assert path.exists()
    text = path.read_text().lower()
    assert "data-layer agnostic" in text
    assert "native pyg" in text
    assert "use custom sampler only" in text
