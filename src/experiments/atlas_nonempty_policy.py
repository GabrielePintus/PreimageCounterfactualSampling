"""Isolated experiment; does not change CertCF's production retention policy.

For binary classification, the target region is an L1 ball intersected with
one affine halfspace. Its nonemptiness is characterized by
a @ center + b - gamma + radius * max(abs(a)) >= 0.
The test certifies existence, not positive volume. Multiclass/discrete/extra
input constraints need a joint feasibility solver and are deliberately rejected.
"""
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
from experiments.atlas_certified_alpha_sweep import coverage_stages


def binary_l1_feasibility(centers, radii, matrices, biases, margin):
    centers = np.asarray(centers, dtype=float)
    n, d = centers.shape
    radii = np.asarray(radii, dtype=float)
    matrices, biases = np.asarray(matrices, dtype=float), np.asarray(biases, dtype=float)
    if matrices.shape != (n, 1, d) or biases.shape != (n, 1) or radii.shape != (n,):
        raise ValueError("Only one affine constraint per continuous L1 region is supported")
    if (not all(np.isfinite(x).all() for x in (centers, radii, matrices, biases))
            or np.any(radii < 0) or not np.isfinite(margin)):
        raise ValueError("Invalid region arrays")
    a = matrices[:, 0, :]
    center_slack = np.sum(a * centers, axis=1) + biases[:, 0] - margin
    coordinate = np.argmax(np.abs(a), axis=1)
    slope = a[np.arange(n), coordinate]
    max_slack = center_slack + radii * np.abs(slope)
    witness = centers.copy()
    shifted = center_slack < 0
    witness[np.arange(n)[shifted], coordinate[shifted]] += radii[shifted] * np.sign(slope[shifted])
    return max_slack >= 0, witness, center_slack, max_slack


def build_nonempty(atlas, initial, labels, cfg):
    import torch

    if atlas.norm != 1 or len(atlas.class_labels) != 2 or atlas.input_bounds is not None or atlas.ohe_slices:
        raise ValueError("This experiment only supports unconstrained binary continuous L1 regions")
    chosen, accepted, witnesses, trace = {}, {}, {}, []
    factor = float(cfg["adaptive_eps_shrink_factor"])
    for step in range(int(cfg["adaptive_eps_max_shrinks"]) + 1):
        radii = initial * factor ** step
        raw = atlas._preimage.compute_all_bounds(
            eps_array=radii, norm=1, batch_size=atlas.batch_size, dtype=torch.float32,
            lirpa_method=atlas.lirpa_method, classification_margin=atlas.classification_margin,
            adaptive_eps=False, show_progress=False,
        )
        for label in atlas.class_labels:
            label = int(label)
            bd = raw[label]
            n = len(bd["X"])
            if step == 0:
                chosen[label] = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in bd.items()}
                accepted[label] = np.zeros(n, dtype=bool)
                witnesses[label] = np.zeros_like(bd["X"], dtype=float)
            else:
                np.testing.assert_array_equal(chosen[label]["X"], bd["X"])
            nonempty, witness, center_slack, max_slack = binary_l1_feasibility(
                bd["X"], bd["eps"], bd["lA"], bd["lbias"], atlas.classification_margin)
            pending = ~accepted[label]
            eligible = (step == 0) | (np.asarray(bd["eps"]) >= float(cfg["adaptive_eps_min"]))
            take = pending & nonempty & eligible
            for i in np.flatnonzero(pending):
                trace.append(dict(class_label=label, anchor_index=int(i), step=step,
                                  radius=float(bd["eps"][i]), center_slack=float(center_slack[i]),
                                  max_affine_slack=float(max_slack[i]), nonempty=bool(nonempty[i]),
                                  accepted=bool(take[i])))
            for k, v in bd.items():
                if isinstance(v, np.ndarray) and v.ndim and len(v) == n:
                    chosen[label][k][take] = v[take]
            chosen[label]["adaptive_eps_n_shrinks"][take] = step
            witnesses[label][take] = witness[take]
            accepted[label] |= take
        print(f"  step={step}: accepted {sum(int(v.sum()) for v in accepted.values())}/{len(labels)}", flush=True)
        if all(v.all() for v in accepted.values()):
            break
    for label, bd in chosen.items():
        keep, n = accepted[label], len(accepted[label])
        bd["eps_initial"] = initial[labels == label].copy()
        bd["original_anchor_index"] = np.arange(n)
        bd["nonempty_witness"] = witnesses[label]
        for k, v in list(bd.items()):
            if isinstance(v, np.ndarray) and v.ndim and len(v) == n:
                bd[k] = v[keep]
        bd["attempted_anchor_count"] = n
        bd["discarded_anchor_count"] = int((~keep).sum())
    atlas.bounds = chosen
    return pd.DataFrame(trace)


def run(config_path):
    import torch
    from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner
    from experiments.network_complexity import sha256_file

    config = yaml.safe_load(Path(config_path).read_text())
    source = TopKHeuristicAblationRunner.from_yaml(config["source_config"])
    if not source._valid_build():
        raise RuntimeError("Source atlas does not match its saved provenance")
    original_hashes = source._atlas_file_hashes()
    coverage_dir = Path(config["coverage_dir"])
    cached_meta = json.loads((coverage_dir / "membership_metadata.json").read_text())
    cache_path = coverage_dir / "membership.npz"
    if (cached_meta["signature"]["source_run"] != source._run_fingerprint()
            or cached_meta["signature"]["atlas_hashes"] != original_hashes
            or cached_meta["cache_sha256"] != sha256_file(cache_path)):
        raise RuntimeError("Stale prediction cache")
    with np.load(cache_path, allow_pickle=False) as data:
        pred = data["predictions"].copy()
        test_indices = data["test_indices"].copy()
    with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
        points = data["x_test"].astype(float)
    np.testing.assert_array_equal(test_indices, np.arange(len(points)))
    with np.load(source.paths.atlas_support, allow_pickle=False) as data:
        anchor_x, anchor_y = data["x_support"].copy(), data["y_support"].copy()
    model = source.source_runner._load_model(source.depth, source.width, source.device)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha in config["alphas"]:
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be in (0, 1]")
        case = root / f"alpha_{alpha:.2f}"
        case.mkdir(parents=True, exist_ok=True)
        signature = dict(source_run=source._run_fingerprint(), source_hashes=original_hashes,
                         config=config, alpha=alpha, cache_sha256=sha256_file(cache_path),
                         code_hashes={str(p): sha256_file(p) for p in [Path(__file__),
                             Path("src/certcf/certification/lirpa.py"), Path("src/certcf/atlas.py"),
                             Path("src/experiments/atlas_coverage.py")]})
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        completed = case / "completed.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["fingerprint"] != fingerprint or any(
                not (case / p).exists() or sha256_file(case / p) != h for p, h in saved["files"].items()
            ):
                raise RuntimeError(f"Stale artifacts at {case}; use a new output directory")
            rows.extend(saved["rows"])
            print(f"Reusing nonempty alpha={alpha}", flush=True)
            continue
        changed = deepcopy(source.config)
        changed["certcf"]["eps_alpha"] = float(alpha)
        variant = TopKHeuristicAblationRunner(changed)
        atlas = variant._new_atlas(model, anchor_x, anchor_y)
        initial = atlas.eps_strategy.compute_eps(anchor_x, anchor_y, norm=1)
        print(f"Nonempty policy alpha={alpha}", flush=True)
        started = time.perf_counter()
        trace = build_nonempty(atlas, initial, anchor_y, changed["certcf"])
        seconds = time.perf_counter() - started
        atlas.save_bounds(case / "atlas")
        trace.to_csv(case / "attempts.csv", index=False)
        class_rows, payload = [], {}
        for label in atlas.class_labels:
            label = int(label)
            bd = atlas.bounds[label]
            m, balls = membership_matrix(atlas, points, label, **config["membership"])
            if m[pred != label].any():
                raise AssertionError("Certified membership contradicts model predictions")
            witness = bd["nonempty_witness"]
            for i, x in enumerate(witness):
                if not atlas._polytope_membership_for_anchor(x, bd, i, delta=0,
                       robust_norm=1, tol=config["membership"]["tolerance"])[0]:
                    raise AssertionError("Feasibility witness fails production membership")
            with torch.inference_mode():
                if len(witness) and not np.all(model(torch.as_tensor(witness, dtype=torch.float32,
                        device=source.device)).argmax(1).cpu().numpy() == label):
                    raise AssertionError("Feasibility witness is not target-valid")
            initial_balls = cdist(points, anchor_x[anchor_y == label].astype(float), metric="cityblock") <= (
                initial[anchor_y == label] + config["membership"]["tolerance"])
            mask = pred == label
            stages = coverage_stages(initial_balls[mask].any(axis=1), balls[mask].any(axis=1), m[mask].any(axis=1))
            centers = np.asarray(bd["adaptive_eps_center_slack"]) >= -config["membership"]["tolerance"]
            row = dict(alpha=float(alpha), class_label=label, n_retained=len(bd["X"]),
                       n_excluding_anchor=int((~centers).sum()),
                       median_final_radius=float(np.median(bd["eps"])) if len(witness) else None,
                       median_shrinks=float(np.median(bd["adaptive_eps_n_shrinks"])) if len(witness) else None,
                       build_seconds=seconds, **stages)
            print(row, flush=True)
            class_rows.append(row)
            payload[f"membership_{label}"] = m
            payload[f"final_balls_{label}"] = balls
        np.savez_compressed(case / "membership.npz", predictions=pred, test_indices=test_indices, **payload)
        files = [case / "membership.npz", case / "attempts.csv", case / "atlas" / "manifest.json"]
        files += [case / "atlas" / f"class_{int(c)}.npz" for c in atlas.class_labels]
        completed.write_text(json.dumps(dict(signature=signature, fingerprint=fingerprint, rows=class_rows,
            files={str(p.relative_to(case)): sha256_file(p) for p in files}), indent=2) + "\n")
        rows.extend(class_rows)
    if original_hashes != source._atlas_file_hashes():
        raise AssertionError("Original atlas changed")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "nonempty_coverage.csv", index=False)
    baseline = pd.read_csv(Path(config["baseline_dir"]) / "coverage_stages.csv")
    comparison = baseline.merge(frame, on=["alpha", "class_label"], suffixes=("_anchor", "_nonempty"), validate="one_to_one")
    comparison.to_csv(root / "policy_comparison.csv", index=False)
    return dict(output_dir=str(root), results=rows)
