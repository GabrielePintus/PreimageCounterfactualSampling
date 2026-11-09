"""Geometric coverage of uncertified initial balls at fixed anchor locations."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist


def entry_thresholds(distances, clearances, *, tolerance=0.0):
    """Smallest alpha with d(x, anchor) <= alpha * clearance + tolerance.

    Handles zero-clearance anchors explicitly; no regions gives infinity.
    """
    distances = np.asarray(distances, dtype=float)
    clearances = np.asarray(clearances, dtype=float)
    if (distances.ndim != 2 or clearances.shape != (distances.shape[1],)
            or not np.isfinite(distances).all() or not np.isfinite(clearances).all()
            or np.any(distances < 0) or np.any(clearances < 0)
            or tolerance < 0 or not np.isfinite(tolerance)):
        raise ValueError("Expected finite nonnegative distances, clearances and tolerance")
    if not len(clearances):
        return np.full(len(distances), np.inf)
    numerator = np.maximum(distances - tolerance, 0.0)
    ratios = np.full_like(distances, np.inf)
    np.divide(numerator, clearances[None, :], out=ratios, where=clearances[None, :] > 0)
    ratios[numerator == 0] = 0.0
    return ratios.min(axis=1)


def plot_sweep(frame, original_alpha, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, group in frame.groupby("class_label"):
        ax.plot(group.alpha, group.coverage * 100, "o-", markersize=3,
                label=f"Class {label} (n={int(group.n_test.iloc[0])})")
    ax.axvline(original_alpha, color="0.4", linestyle="--", linewidth=1,
               label=f"Saved atlas initialization: α={original_alpha:g}")
    ax.set(xlabel="Initial-radius scale α", ylabel="Held-out coverage (%)",
           title="Initial balls only — no certification or shrinkage", ylim=(-2, 102))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path):
    from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner
    from experiments.network_complexity import sha256_file

    config = yaml.safe_load(Path(config_path).read_text())
    source = TopKHeuristicAblationRunner.from_yaml(config["source_config"])
    if not source._valid_build():
        raise RuntimeError("Existing source atlas is missing or stale; no rebuilding is allowed")
    if source.config["certcf"]["norm"] != 1:
        raise ValueError("This pilot adapter expects L1 geometry")
    alphas = np.unique(np.asarray(config["alphas"], dtype=float))
    if not len(alphas) or not np.isfinite(alphas).all() or np.any(alphas <= 0) or np.any(alphas > 1):
        raise ValueError("Alphas must be in the implemented strategy range (0, 1]")
    coverage_dir = Path(config["coverage_dir"])
    cache_path = coverage_dir / "membership.npz"
    metadata = json.loads((coverage_dir / "membership_metadata.json").read_text())
    if (metadata["signature"]["source_run"] != source._run_fingerprint()
            or metadata["signature"]["atlas_hashes"] != source._atlas_file_hashes()
            or metadata["cache_sha256"] != sha256_file(cache_path)):
        raise RuntimeError("Coverage cache does not match the existing source artifacts")
    with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
        x_test = data["x_test"].astype(float)
    with np.load(cache_path, allow_pickle=False) as cached:
        predicted = cached["predictions"].copy()
        labels = cached["class_labels"].copy()
        np.testing.assert_array_equal(cached["test_indices"], np.arange(len(x_test)))
    with np.load(source.paths.atlas_support, allow_pickle=False) as support:
        reference_x, reference_y = support["x_support"].astype(float), support["y_support"]
    manifest = json.loads(source.paths.atlas_manifest.read_text())
    alpha_original = float(source.config["certcf"]["eps_alpha"])
    rows, thresholds, diagnostics, arrays = [], [], {}, {}
    for label in labels:
        label = int(label)
        with np.load(source.paths.atlas_dir / manifest["files"][str(label)], allow_pickle=False) as bounds:
            centers = bounds["X"].astype(float)
            initial = bounds["eps_initial"].copy()
            final = bounds["eps"].copy()
        # Match the saved construction: nearest opposite among SELECTED anchors.
        clearance = cdist(centers, reference_x[reference_y != label], metric="cityblock").min(axis=1)
        np.testing.assert_allclose(initial, alpha_original * clearance, rtol=1e-7, atol=1e-8)
        indices = np.flatnonzero(predicted == label)
        distances = cdist(x_test[indices], centers, metric="cityblock")
        critical = entry_thresholds(distances, clearance)
        critical_tol = entry_thresholds(distances, clearance, tolerance=float(config["tolerance"]))
        arrays[f"distances_{label}"] = distances
        arrays[f"clearances_{label}"] = clearance
        arrays[f"test_indices_{label}"] = indices
        thresholds.append(pd.DataFrame(dict(test_index=indices, class_label=label,
                                            entry_alpha=critical, entry_alpha_with_tolerance=critical_tol)))
        counts = []
        for alpha in alphas:
            covered = (distances <= alpha * clearance + config["tolerance"]).any(axis=1)
            np.testing.assert_array_equal(covered, critical_tol <= alpha)
            count = int(covered.sum())
            counts.append(count)
            rows.append(dict(class_label=label, alpha=float(alpha), n_regions=len(centers),
                             n_test=len(indices), n_covered=count,
                             coverage=count / len(indices) if len(indices) else np.nan))
        if np.any(np.diff(counts) < 0):
            raise AssertionError("Coverage must be monotone for fixed centers and clearances")
        diagnostics[str(label)] = dict(
            n_test=len(indices), n_regions=len(centers),
            first_entry_alpha=float(critical.min()) if len(indices) else None,
            median_entry_alpha=float(np.median(critical)) if len(indices) else None,
            max_entry_alpha=float(critical.max()) if len(indices) else None,
            initial_hits=int((distances <= initial + config["tolerance"]).any(axis=1).sum()),
            final_ball_hits=int((distances <= final + config["tolerance"]).any(axis=1).sum()),
        )
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "initial_ball_coverage.csv", index=False)
    pd.concat(thresholds, ignore_index=True).to_csv(output / "entry_alpha_by_point.csv", index=False)
    np.savez_compressed(output / "distances.npz", **arrays)
    plot_sweep(frame, alpha_original, output / "initial_ball_coverage")
    report = dict(config=config, original_alpha=alpha_original, diagnostics=diagnostics,
                  source_run=source._run_fingerprint(), atlas_hashes=source._atlas_file_hashes(),
                  coverage_cache_sha256=sha256_file(cache_path), code_sha256=sha256_file(Path(__file__)),
                  interpretation="Uncertified initial-ball coverage; no model evaluation, LiRPA, shrinkage or erosion.",
                  reference_set="The same 1000 selected anchors used in the saved atlas construction.")
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return dict(output_dir=str(output), diagnostics=diagnostics,
                selected_results=frame[frame.alpha.isin([.2, .5, .6, .7, .8, .9, 1.])].to_dict("records"))
