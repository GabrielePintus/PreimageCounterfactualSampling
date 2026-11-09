"""Explicit training-box PREMAP pilot; does not overwrite source experiments."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import torch
from torch import nn
import yaml

from certcf import BuildRequest, CertCFAtlas, InputDomain, RegionGeometry
from certcf.certification.premap import PremapBackend, PremapConfig


def training_box(points):
    """Only training inputs determine this domain; never expand from test data."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected a nonempty finite training matrix")
    lo, hi = points.min(0), points.max(0)
    if np.any(lo >= hi):
        raise ValueError("PREMAP pilot needs nonconstant numerical dimensions")
    return RegionGeometry(np.empty((0, points.shape[1])), np.empty(0), lo, hi)


def explicit_graph(model, example):
    """Materialize this known TabularClassifier's sequential inference graph.

    Loaded LazyLinear layers may retain in_features=0 despite initialized weights.
    Recreate their exact affine maps from tensor shapes and check model outputs.
    No graph inference for arbitrary wrapper types is attempted.
    """
    from models.classifiers import TabularClassifier
    if type(model) is nn.Sequential:
        network = model
    elif type(model) is TabularClassifier and all(t == "numerical" for t in model.input_types):
        network = model.net
    else:
        raise TypeError("Pilot requires a numerical TabularClassifier or explicit Sequential")
    layers = []
    model.eval()
    with torch.no_grad():
        expected = model(example)
        for layer in network:
            if isinstance(layer, nn.Linear):
                replacement = nn.Linear(layer.weight.shape[1], layer.weight.shape[0], bias=layer.bias is not None)
                replacement.weight.copy_(layer.weight)
                if layer.bias is not None:
                    replacement.bias.copy_(layer.bias)
                layers.append(replacement)
            elif type(layer) is nn.ReLU:
                layers.append(nn.ReLU())
            elif type(layer) in (nn.Dropout, nn.Identity):
                continue
            else:
                raise TypeError(f"Unsupported layer {type(layer)}")
        graph = nn.Sequential(*layers).eval()
        torch.testing.assert_close(graph(example), expected, rtol=0, atol=0)
    return graph


def region_feasibility(region, model, margin):
    g = region.geometry
    solved = linprog(np.zeros(g.dimension), A_ub=-g.A, b_ub=g.b,
                     bounds=list(zip(g.lower, g.upper)), method="highs")
    if solved.status == 2:
        return {"region_id": region.region_id, "feasible": False}
    if not solved.success:
        raise RuntimeError(f"Region feasibility solve failed: {solved.message}")
    if not g.contains(solved.x, tolerance=1e-6):
        raise AssertionError("Feasibility witness violates region geometry")
    with torch.no_grad():
        scores = model(torch.tensor(solved.x, dtype=torch.float32).unsqueeze(0))[0].numpy()
    actual = float(scores[region.target_class] - scores[1 - region.target_class])
    if actual < margin - 1e-5:
        raise AssertionError(f"Region witness contradicts target margin: {actual}")
    return {"region_id": region.region_id, "feasible": True,
            "model_margin_at_witness": actual, "witness": solved.x.tolist()}


def run(config_path, python, repository):
    from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner
    from experiments.network_complexity import sha256_file
    from experiments.atlas_coverage import row_overlap_count

    config = yaml.safe_load(Path(config_path).read_text())
    if config["domain_policy"] != "training_minmax_box":
        raise ValueError("Unsupported domain policy")
    torch.set_num_threads(1)
    source = TopKHeuristicAblationRunner.from_yaml(config["source_config"])
    if not source._valid_build():
        raise RuntimeError("Existing source atlas is missing or stale; will not retrain/rebuild")
    original_hashes = source._atlas_file_hashes()
    with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
        train, test = data["x_train"].copy(), data["x_test"].copy()
        truth = data["y_test"].copy()
        validation = data["x_validation"].copy()
    if row_overlap_count(test, train) or row_overlap_count(test, validation):
        raise AssertionError("Test split overlaps training/validation")
    model = source.source_runner._load_model(source.depth, source.width, "cpu")
    graph = explicit_graph(model, torch.tensor(train[:128], dtype=torch.float32))
    with torch.no_grad():
        logits = graph(torch.tensor(test, dtype=torch.float32)).numpy()
        np.testing.assert_array_equal(logits, model(torch.tensor(test, dtype=torch.float32)).numpy())
    predictions = logits.argmax(1)
    box = training_box(train)
    labels = (0, 1)
    margin = float(source.config["certcf"]["classification_margin"])
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), Path("src/certcf/atlas.py"),
                  Path("src/certcf/certification/premap.py"), Path("src/certcf/certification/premap_worker.py"),
                  Path("src/certcf/geometry/regions.py")]
    identity = {"config": config, "source_run": source._run_fingerprint(),
                "source_atlas_hashes": original_hashes,
                "code": {str(p): sha256_file(p) for p in code_paths},
                "python": str(Path(python).resolve()), "repository": str(Path(repository).resolve())}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Baseline is measured on exactly the same held-out points and tolerance.
    baseline = CertCFAtlas.from_directory(source.paths.atlas_dir, classification_margin=margin)
    baseline_counts = {c: int(baseline.membership_matrix(test[predictions == c], c,
                               **config["membership"]).any(1).sum()) for c in labels}
    metadata = {"identity": identity, "fingerprint": fingerprint,
                "n_train": len(train), "n_test": len(test), "input_dim": test.shape[1],
                "depth": source.depth, "width": source.width,
                "n_parameters": sum(p.numel() for p in graph.parameters()),
                "test_accuracy": float(np.mean(predictions == truth)), "margin": margin,
                "domain_lower": box.lower.tolist(), "domain_upper": box.upper.tolist(),
                "original_atlas_covered": baseline_counts,
                "note": "Exploratory global training-box construction; differs from original local L1 domains. Sampled stopping coverage is not held-out coverage."}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    for budget in config["premap"]["region_budgets"]:
        for label in labels:
            case = root / f"regions_{budget:03d}" / f"class_{label}"
            completed = case / "completed.json"
            if completed.exists():
                saved = json.loads(completed.read_text())
                if saved["fingerprint"] != fingerprint or any(
                    not (case / p).exists() or sha256_file(case / p) != h for p, h in saved["files"].items()
                ):
                    raise RuntimeError(f"Stale artifacts in {case}; choose a new output directory")
                rows.append(saved["row"])
                continue
            case.mkdir(parents=True, exist_ok=True)
            settings = {k: v for k, v in config["premap"].items() if k != "region_budgets"}
            backend = PremapBackend(PremapConfig(python=python, repository=repository, max_regions=int(budget), **settings))
            request = BuildRequest(graph, (InputDomain(f"training-box-class-{label}", label, box),), labels, (test.shape[1],), margin)
            print(f"PREMAP class={label}, region budget={budget}, timeout={settings['timeout_s']}s...", flush=True)
            started = time.perf_counter()
            built = backend.build(request)
            elapsed = time.perf_counter() - started
            atlas = CertCFAtlas.from_regions(built, distance_norm=1)
            atlas.save_bounds(case / "atlas")
            (case / "worker.log").write_text(built.diagnostics.get("log_tail", ""))
            memberships = atlas.membership_matrix(test, label, **config["membership"])
            strict = atlas.membership_matrix(test, label, tolerance=0, batch_size=config["membership"]["batch_size"])
            if memberships[predictions != label].any():
                raise AssertionError("Certified membership contradicts classifier predictions")
            witnesses = [region_feasibility(r, graph, margin) for r in built.regions]
            (case / "region_feasibility.json").write_text(json.dumps(witnesses, indent=2) + "\n")
            eligible = predictions == label
            inside = box.contains(test, tolerance=config["membership"]["tolerance"])
            covered = memberships.any(1)
            if np.any(covered & ~inside):
                raise AssertionError("Coverage escapes source domain")
            query_ids = np.random.default_rng(settings["seed"] + label).permutation(np.flatnonzero(~eligible))[:config["queries_per_target"]]
            query_records = []
            for query_id in query_ids:
                start = time.perf_counter()
                cf = atlas.find_counterfactual(test[query_id], label, method="sorted")
                seconds = time.perf_counter() - start
                entry = {"test_index": int(query_id), "success": bool(cf.success), "seconds": seconds}
                if cf.success:
                    with torch.no_grad():
                        scores = graph(torch.tensor(cf.x_cf, dtype=torch.float32).unsqueeze(0))[0].numpy()
                    actual_margin = float(scores[label] - scores[1 - label])
                    if actual_margin < margin - 1e-5:
                        raise AssertionError("Projected CF contradicts target margin")
                    entry.update(x_cf=cf.x_cf.tolist(), l1_distance=float(cf.distance), model_margin=actual_margin)
                query_records.append(entry)
            (case / "queries.json").write_text(json.dumps(query_records, indent=2) + "\n")
            np.savez_compressed(case / "membership.npz", test_indices=np.arange(len(test)), predictions=predictions,
                                membership=memberships, strict_membership=strict, inside_domain=inside)
            stats = built.diagnostics.get("domain_stats", [])
            row = {"region_budget": int(budget), "class_label": label, "build_seconds": elapsed,
                   "status": built.status, "timed_out": bool(built.diagnostics.get("timed_out")),
                   "n_regions": len(built.regions), "n_feasible_regions": sum(w["feasible"] for w in witnesses),
                   "n_test": int(eligible.sum()), "n_inside_domain": int((eligible & inside).sum()),
                   "n_covered": int((eligible & covered).sum()),
                   "n_covered_strict": int(strict[eligible].any(1).sum()),
                   "held_out_coverage": float(covered[eligible].mean()),
                   "coverage_conditional_on_domain": float(covered[eligible & inside].mean()) if (eligible & inside).any() else None,
                   "internal_uniform_coverage": stats[0].get("estimated_coverage") if stats else None,
                   "queries_attempted": len(query_records), "queries_successful": sum(q["success"] for q in query_records),
                   "mean_query_seconds": float(np.mean([q["seconds"] for q in query_records]))}
            files = [case / "membership.npz", case / "region_feasibility.json", case / "queries.json",
                     case / "atlas" / "manifest.json"]
            manifest = json.loads((case / "atlas" / "manifest.json").read_text())
            files.append(case / "atlas" / manifest["file"])
            completed.write_text(json.dumps({"fingerprint": fingerprint, "row": row,
                "files": {str(p.relative_to(case)): sha256_file(p) for p in files}}, indent=2) + "\n")
            rows.append(row)
            pd.DataFrame(rows).to_csv(root / "coverage.csv", index=False)
            print(json.dumps(row), flush=True)
    if source._atlas_file_hashes() != original_hashes:
        raise AssertionError("Source atlas changed")
    pd.DataFrame(rows).to_csv(root / "coverage.csv", index=False)
    return {"output_dir": str(root), "results": rows}
