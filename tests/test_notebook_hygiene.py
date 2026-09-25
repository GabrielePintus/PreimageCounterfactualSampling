from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_notebooks_do_not_contain_execution_outputs() -> None:
    problems: list[str] = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None:
                problems.append(f"{path.name}: cell {index} has an execution count")
            if cell.get("outputs"):
                problems.append(f"{path.name}: cell {index} has outputs")
    assert not problems, "\n".join(problems)


def test_notebooks_do_not_contain_absolute_home_paths() -> None:
    offenders = [
        path.name
        for path in sorted((ROOT / "notebooks").glob("*.ipynb"))
        if "/home/" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"Absolute home paths found in: {offenders}"
