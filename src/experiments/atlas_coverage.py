"""Held-out data coverage, not geometric volume or counterfactual success.

Membership and nested-prefix analysis are reusable for other datasets. The
current adapter only reads the existing synthetic-32 top-k experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def membership_matrix(atlas, points, label, *, tolerance=1e-6, batch_size=256):
    """Use production constraints, vectorizing points within each region.

    Return certified membership and the enclosing-ball membership diagnostic.
    Columns use the exact order of the saved atlas regions; no retrieval occurs.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite 2D array")
    if batch_size < 1 or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("invalid membership batch size or tolerance")
    bounds = atlas.bounds[int(label)]
    centers = np.asarray(bounds["X"], dtype=np.float64)
    membership = np.zeros((len(points), len(centers)), dtype=bool)
    balls = np.zeros_like(membership)
    for i, center in enumerate(centers):
        a, b, _, radius = atlas._erode_constraints(
            bounds["lA"][i], bounds["lbias"][i], center, points.shape[1],
            0.0, atlas.norm, float(bounds["eps"][i]),
        )
        if a is None:
            continue
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite certified constraints")
        for start in range(0, len(points), batch_size):
            x = points[start:start + batch_size]
            inside_ball = np.linalg.norm(x - center, ord=atlas.norm, axis=1) <= radius + tolerance
            inside = inside_ball & np.all(x @ a.T + b >= -tolerance, axis=1)
            if atlas.input_bounds is not None:
                lower, upper = atlas.input_bounds
                inside &= np.all((x >= lower - tolerance) & (x <= upper + tolerance), axis=1)
            balls[start:start + len(x), i] = inside_ball
            membership[start:start + len(x), i] = inside
    return membership, balls


def nested_curves(membership, *, fractions, orderings, seed, label):
    """Coverage conditional on predicted class; empty denominators are NaN."""
    membership = np.asarray(membership, dtype=bool)
    fractions = np.asarray(fractions, dtype=float)
    if membership.ndim != 2 or not len(fractions) or orderings < 1:
        raise ValueError("invalid analysis inputs")
    if not np.isfinite(fractions).all() or np.any(fractions <= 0) or np.any(fractions > 1):
        raise ValueError("fractions must lie in (0, 1]")
    fractions = np.unique(np.append(fractions, 1.0))
    n_points, n_regions = membership.shape
    rows, permutations = [], []
    for repeat in range(orderings):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(label), repeat]))
        order = rng.permutation(n_regions)
        permutations.append(order)
        covered = np.zeros(n_points, dtype=bool)
        previous_k, previous_count = 0, 0
        for fraction in fractions:
            k = min(n_regions, math.ceil(float(fraction) * n_regions))
            if k > previous_k:
                covered |= membership[:, order[previous_k:k]].any(axis=1)
            count = int(covered.sum())
            if count < previous_count:
                raise AssertionError("coverage decreased along a nested prefix")
            if k == n_regions and count != int(membership.any(axis=1).sum()):
                raise AssertionError("full-atlas endpoint mismatch")
            rows.append(dict(class_label=int(label), ordering=repeat, fraction=float(fraction),
                             n_regions=k, n_test=n_points, n_covered=count,
                             coverage=count / n_points if n_points else np.nan))
            previous_k, previous_count = k, count
    return pd.DataFrame(rows), np.asarray(permutations)


def row_overlap_count(first, second):
    """Count first-array rows also present in second (exact stored inputs)."""
    known = {tuple(row) for row in np.asarray(second)}
    return sum(tuple(row) in known for row in np.asarray(first))


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(summary.class_label.unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(3.7 * len(labels), 3.3), squeeze=False, sharey=True)
    for ax, label in zip(axes[0], labels):
        group = summary[summary.class_label == label].sort_values("n_regions")
        x = group.n_regions.to_numpy()
        mean = 100 * group.coverage_mean.to_numpy()
        ax.fill_between(x, 100 * group.coverage_min.to_numpy(), 100 * group.coverage_max.to_numpy(), alpha=.2)
        ax.plot(x, mean, "o-", markersize=4)
        n_test = int(group.n_test.iloc[0])
        ax.set_title(f"Predicted class {label} · n={n_test}")
        ax.set_xlabel("Number of certified regions")
        ax.set_ylim(-1, 101)
        ax.grid(alpha=.2)
        ax.text(.04, .93, f"Full atlas: {mean[-1]:.1f}%", transform=ax.transAxes, va="top")
    axes[0, 0].set_ylabel("Held-out coverage (%)")
    fig.text(.5, .015, "Line: mean; band: min–max across nested random orderings (not a confidence interval).", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path, *, force=False):
    """Read the source artifacts only; never prepare, train or build them."""
    import torch
    from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner
    from experiments.network_complexity import sha256_file

    config = yaml.safe_load(Path(config_path).read_text())
    source = TopKHeuristicAblationRunner.from_yaml(config["source_config"])
    if not source._valid_build():
        raise RuntimeError("Source atlas is missing or stale; refusing to rebuild it automatically")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "source_run": source._run_fingerprint(),
        "atlas_hashes": source._atlas_file_hashes(),
        "membership": config["membership"],
        "delta": 0.0,
        "evaluation_split": "entire x_test; grouped by model prediction",
        "code_hashes": {str(p): sha256_file(p) for p in [
            Path(__file__), Path("src/certcf/atlas.py"),
            Path("src/experiments/topk_heuristic_ablation.py"),
            Path("src/experiments/network_complexity.py"),
        ]},
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_path, metadata_path = output / "membership.npz", output / "membership_metadata.json"
    reusable = False
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text())
        reusable = (metadata.get("fingerprint") == fingerprint and
                    metadata.get("cache_sha256") == sha256_file(cache_path))
    if not reusable:
        print("Loading the existing synthetic-32 model and atlas...", flush=True)
        atlas, model, x_test, y_test, _ = source._load_atlas_and_data()
        with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
            x_train = data["x_train"]
            x_validation = data["x_validation"]
            dataset_metadata = json.loads(str(data["metadata_json"].item()))
        overlaps = {"test_train": row_overlap_count(x_test, x_train),
                    "test_validation": row_overlap_count(x_test, x_validation)}
        for label in atlas.class_labels:
            anchors = atlas.bounds[int(label)]["X"]
            overlaps[f"test_anchors_{label}"] = row_overlap_count(x_test, anchors)
            if row_overlap_count(anchors, x_train) != len(anchors):
                raise AssertionError("An atlas anchor is not in the training split")
        if any(overlaps.values()):
            raise AssertionError(f"Exact row overlaps in evaluation split: {overlaps}")
        model.eval()
        with torch.inference_mode():
            predictions = model(torch.as_tensor(x_test, device=source.device)).argmax(1).cpu().numpy()
        if not np.isin(predictions, atlas.class_labels).all():
            raise AssertionError("Predicted class missing from atlas")
        payload = dict(test_indices=np.arange(len(x_test)), predictions=predictions, true_labels=y_test,
                       class_labels=np.asarray(atlas.class_labels))
        diagnostics = {}
        for label in atlas.class_labels:
            label = int(label)
            membership, balls = membership_matrix(atlas, x_test, label, **config["membership"])
            bd = atlas.bounds[label]
            # Independent scalar check for every saved anchor (positive controls).
            anchor_checks = [atlas._polytope_membership_for_anchor(
                np.asarray(center, dtype=np.float64), bd, i, delta=0, robust_norm=atlas.norm,
                tol=config["membership"]["tolerance"],
            )[0] for i, center in enumerate(bd["X"])]
            if not all(anchor_checks):
                raise AssertionError("Saved region does not contain its anchor")
            # Check vectorized positive controls too; held-out membership can be all false.
            anchor_matrix, _ = membership_matrix(atlas, bd["X"], label, **config["membership"])
            if not np.diag(anchor_matrix).all():
                raise AssertionError("Vectorized anchor membership mismatch")
            rng = np.random.default_rng(42 + label)
            for _ in range(100):
                row, col = int(rng.integers(len(x_test))), int(rng.integers(len(bd["X"])))
                expected = atlas._polytope_membership_for_anchor(
                    x_test[row].astype(np.float64), bd, col, delta=0, robust_norm=atlas.norm,
                    tol=config["membership"]["tolerance"],
                )[0]
                if membership[row, col] != expected:
                    raise AssertionError("Vectorized/scalar membership mismatch")
            if membership[predictions != label].any():
                raise AssertionError("A certified region covers an input predicted as another class")
            payload[f"membership_{label}"] = membership
            payload[f"balls_{label}"] = balls
            mask = predictions == label
            diagnostics[str(label)] = dict(
                n_regions=len(bd["X"]), n_test=int(mask.sum()),
                n_covered=int(membership[mask].any(axis=1).sum()),
                n_in_final_balls=int(balls[mask].any(axis=1).sum()),
                anchors_verified=int(sum(anchor_checks)),
                median_radius=float(np.median(bd["eps"])),
            )
            print(f"Class {label}: {diagnostics[str(label)]}", flush=True)
        np.savez_compressed(cache_path, **payload)
        metadata = dict(fingerprint=fingerprint, signature=signature,
                        cache_sha256=sha256_file(cache_path), overlaps=overlaps,
                        dataset_metadata=dataset_metadata, diagnostics=diagnostics,
                        test_accuracy=float(np.mean(predictions == y_test)))
        _json_write(metadata_path, metadata)
    else:
        print("Reusing cached predictions and membership matrices.", flush=True)

    frames, permutations = [], {}
    with np.load(cache_path, allow_pickle=False) as cached:
        predictions = cached["predictions"]
        for label in cached["class_labels"]:
            label = int(label)
            matrix = cached[f"membership_{label}"][predictions == label]
            frame, orders = nested_curves(matrix, label=label, **config["analysis"])
            frames.append(frame)
            permutations[f"orders_{label}"] = orders
    curves = pd.concat(frames, ignore_index=True)
    summary = curves.groupby(["class_label", "fraction", "n_regions"], as_index=False).agg(
        n_test=("n_test", "first"), n_covered_mean=("n_covered", "mean"),
        coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
    )
    curves.to_csv(output / "coverage_by_ordering.csv", index=False)
    summary.to_csv(output / "coverage_summary.csv", index=False)
    full = curves[(curves.fraction == 1) & (curves.ordering == 0)]
    full[["class_label", "n_regions", "n_test", "n_covered", "coverage"]].to_csv(output / "coverage_full.csv", index=False)
    np.savez_compressed(output / "orderings.npz", **permutations)
    _plot(summary, output / "coverage_curve")
    _json_write(output / "analysis_metadata.json", dict(config=config, membership_fingerprint=fingerprint,
                                                       variability="random region orderings, not test-sample uncertainty"))
    report = {"output_dir": str(output), "reused_membership": reusable,
              "full_atlas": metadata["diagnostics"], "test_accuracy": metadata["test_accuracy"]}
    return report
