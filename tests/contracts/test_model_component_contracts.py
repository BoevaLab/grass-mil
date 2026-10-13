from pathlib import Path


def test_contract_markdown_exists_and_has_required_sections():
    path = Path("docs/model_component_contracts.md")
    assert path.exists()
    text = path.read_text()
    assert "Graph Unit Contract" in text
    assert "Loss Contracts" in text
    assert "Sampler Contracts" in text
