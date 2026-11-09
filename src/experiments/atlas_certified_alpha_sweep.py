"""Compare initial balls, final balls and certified regions for fixed anchors."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import yaml

from experiments.atlas_coverage import membership_matrix


def coverage_stages(initial, final, certified):
    """Count disjoint losses; masks describe unions, not individual regions."""
    initial, final, certified = [np.asarray(x, dtype=bool) for x in (initial, final, certified)]
    if initial.ndim != 1 or initial.shape != final.shape or initial.shape != certified.shape:
        raise ValueError("Stage masks must be aligned one-dimensional arrays")
    if np.any(final & ~initial) or np.any(certified & ~final):
        raise AssertionError("Coverage sets are not nested")
    n = len(initial)
    return dict(n_test=n, initial_covered=int(initial.sum()), final_ball_covered=int(final.sum()),
                certified_covered=int(certified.sum()), outside_initial=int((~initial).sum()),
                lost_shrink_or_discard=int((initial & ~final).sum()),
                lost_affine_constraints=int((final & ~certified).sum()),
                initial_coverage=float(initial.mean()) if n else None,
                final_ball_coverage=float(final.mean()) if n else None,
                certified_coverage=float(certified.mean()) if n else None)


def plot_comparison(frame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for ax, (label, group) in zip(axes, frame.groupby("class_label")):
        group = group.sort_values("alpha")
        x = np.arange(len(group))
        ax.bar(x - .24, group.initial_coverage * 100, width=.24, label="Initial balls")
        ax.bar(x, group.final_ball_coverage * 100, width=.24, label="After shrinkage")
        ax.bar(x + .24, group.certified_coverage * 100, width=.24, label="Certified regions")
        ax.set_xticks(x, [f"{a:g}" for a in group.alpha])
        ax.set(xlabel="Initial-radius scale α", title=f"Predicted class {label}", ylim=(0, 105))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.text(.03, .96, "Final-ball and certified coverage: 0%" if (group.final_ball_covered == 0).all()
                else "", transform=ax.transAxes, va="top", fontsize=9)
    axes[0].set_ylabel("Held-out coverage (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path):
    from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner
    from experiments.network_complexity import sha256_file

    config = yaml.safe_load(Path(config_path).read_text())
    source = TopKHeuristicAblationRunner.from_yaml(config["source_config"])
    if not source._valid_build():
        raise RuntimeError("Original source atlas is missing or stale")
    if source.config["certcf"]["norm"] != 1:
        raise ValueError("This pilot adapter requires L1")
    original_hashes = source._atlas_file_hashes()
    coverage_dir = Path(config["coverage_dir"])
    cache_path = coverage_dir / "membership.npz"
    meta = json.loads((coverage_dir / "membership_metadata.json").read_text())
    if (meta["signature"]["source_run"] != source._run_fingerprint()
            or meta["signature"]["atlas_hashes"] != original_hashes
            or meta["cache_sha256"] != sha256_file(cache_path)):
        raise RuntimeError("Coverage cache does not match source artifacts")
    with np.load(cache_path, allow_pickle=False) as cached:
        predictions = cached["predictions"].copy()
        indices = cached["test_indices"].copy()
    with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
        points = data["x_test"].astype(float)
    np.testing.assert_array_equal(indices, np.arange(len(points)))
    with np.load(source.paths.atlas_support, allow_pickle=False) as support:
        anchor_x, anchor_y = support["x_support"].copy(), support["y_support"].copy()
    labels = np.unique(anchor_y)
    clearances = {int(c): cdist(anchor_x[anchor_y == c].astype(float),
                              anchor_x[anchor_y != c].astype(float), metric="cityblock").min(axis=1)
                  for c in labels}
    original_alpha = float(source.config["certcf"]["eps_alpha"])
    alphas = np.unique(np.append(np.asarray(config["alphas"], dtype=float), original_alpha))
    if not np.isfinite(alphas).all() or np.any(alphas <= 0) or np.any(alphas > 1):
        raise ValueError("Alpha must be in (0, 1]")
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    model = source.source_runner._load_model(source.depth, source.width, source.device)
    rows = []
    for alpha in alphas:
        case = root / f"alpha_{alpha:.2f}"
        case.mkdir(parents=True, exist_ok=True)
        changed = deepcopy(source.config)
        changed["certcf"]["eps_alpha"] = float(alpha)
        changed["artifacts"]["output_dir"] = str(case)
        variant = TopKHeuristicAblationRunner(changed)
        identity = dict(source_run=source._run_fingerprint(), original_hashes=original_hashes,
                        coverage_sha256=sha256_file(cache_path), config=changed,
                        membership=config["membership"],
                        code={str(p): sha256_file(p) for p in [Path(__file__),
                              Path("src/experiments/atlas_coverage.py"), Path("src/certcf/atlas.py")]})
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        completed = case / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if (saved["fingerprint"] != fingerprint or any(
                not (case / p).exists() or sha256_file(case / p) != h for p, h in saved["files"].items()
            )):
                raise RuntimeError(f"Stale variant artifacts in {case}; choose a new output directory")
            print(f"Reusing completed alpha={alpha:g}", flush=True)
            rows.extend(saved["rows"])
            continue
        atlas = variant._new_atlas(model, anchor_x, anchor_y)
        start = time.perf_counter()
        if alpha == original_alpha:
            atlas.load_bounds(source.paths.atlas_dir)
            build_seconds = None
        else:
            print(f"Building alpha={alpha:g} on the same {len(anchor_x)} anchors...", flush=True)
            atlas.build(verbose=True)
            build_seconds = time.perf_counter() - start
        atlas.save_bounds(case / "atlas")
        payload, class_rows = {}, []
        for label in labels:
            label = int(label)
            bd = atlas.bounds[label]
            centers = np.asarray(bd["X"], dtype=float)
            original_centers = anchor_x[anchor_y == label].astype(float)
            lookup = {tuple(x): i for i, x in enumerate(original_centers)}
            retained = np.array([lookup[tuple(x)] for x in centers], dtype=int)
            np.testing.assert_allclose(bd["eps_initial"], alpha * clearances[label][retained], rtol=1e-7, atol=1e-8)
            if np.any(np.asarray(bd["eps"]) > np.asarray(bd["eps_initial"]) + 1e-6):
                raise AssertionError("Shrinking increased a radius")
            membership, final_balls = membership_matrix(atlas, points, label, **config["membership"])
            initial_balls = cdist(points, original_centers, metric="cityblock") <= (
                alpha * clearances[label] + config["membership"]["tolerance"])
            if membership[predictions != label].any():
                raise AssertionError("Certificate contradicts model predictions")
            positive, _ = membership_matrix(atlas, centers, label, **config["membership"])
            if not np.diag(positive).all():
                raise AssertionError("Saved regions do not contain their anchors")
            for i, center in enumerate(centers):
                if not atlas._polytope_membership_for_anchor(center, bd, i, delta=0,
                       robust_norm=atlas.norm, tol=config["membership"]["tolerance"])[0]:
                    raise AssertionError("Production anchor membership check failed")
            mask = predictions == label
            stage = coverage_stages(initial_balls[mask].any(axis=1),
                                    final_balls[mask].any(axis=1), membership[mask].any(axis=1))
            row = dict(alpha=float(alpha), class_label=label, n_attempted=len(original_centers),
                       n_retained=len(centers), median_initial_radius=float(np.median(alpha * clearances[label])),
                       median_final_radius=float(np.median(bd["eps"])),
                       median_shrinks=float(np.median(bd["adaptive_eps_n_shrinks"])),
                       shrink_fraction=float(np.mean(np.asarray(bd["adaptive_eps_n_shrinks"]) > 0)),
                       build_seconds=build_seconds, **stage)
            print(row, flush=True)
            class_rows.append(row)
            payload[f"membership_{label}"] = membership
            payload[f"initial_balls_{label}"] = initial_balls
            payload[f"final_balls_{label}"] = final_balls
        np.savez_compressed(case / "membership.npz", predictions=predictions, test_indices=indices, **payload)
        files = [case / "membership.npz", case / "atlas" / "manifest.json"]
        files += [case / "atlas" / f"class_{int(c)}.npz" for c in labels]
        completed.write_text(json.dumps(dict(fingerprint=fingerprint, identity=identity, rows=class_rows,
            files={str(p.relative_to(case)): sha256_file(p) for p in files}), indent=2) + "\n")
        rows.extend(class_rows)
    if source._atlas_file_hashes() != original_hashes:
        raise AssertionError("Original atlas artifacts changed")
    frame = pd.DataFrame(rows).sort_values(["class_label", "alpha"])
    frame.to_csv(root / "coverage_stages.csv", index=False)
    plot_comparison(frame, root / "coverage_stages")
    return dict(output_dir=str(root), results=rows)
