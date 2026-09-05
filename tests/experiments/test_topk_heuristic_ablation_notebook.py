from __future__ import annotations

import json
from pathlib import Path


def test_topk_notebook_contains_required_validation_and_analysis():
    path = Path("notebooks/TopKHeuristicAblation.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )

    required = [
        "Protocol and artifact validation",
        "exact_recovery_rate",
        "within_1pct_rate",
        "gap_absolute_p95",
        "strict_runtime_ms_mean",
        "Minimum $k$ required",
        "right-censored",
        "fixed certified atlas",
        "Clopper--Pearson",
        "upper_95_one_sided",
        "bootstrap_mean_interval",
        "Paired incremental comparisons",
        "wilcoxon",
        "Holm-adjusted p-value",
        "Cost--quality trade-off",
        "Worst-case queries",
        "Held-out conformal calibration",
        "recovery_rank_censored",
    ]
    for marker in required:
        assert marker in source

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
