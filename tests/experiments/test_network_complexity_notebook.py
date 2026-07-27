import json
from pathlib import Path


def test_network_complexity_notebook_has_required_analysis_sections():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks" / "NetworkComplexityScaling.ipynb").read_text(encoding="utf-8")
    )
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    ).lower()
    for required in (
        "complete per-architecture table",
        "width × depth heatmaps",
        "build cost versus parameter count",
        "marginal trends by width and depth",
        "empirical certcf build-time complexity",
        "leave-one-architecture-out",
        "layer-width interaction",
        "failures and resource limits",
        "25,000",
    ):
        assert required in sources
