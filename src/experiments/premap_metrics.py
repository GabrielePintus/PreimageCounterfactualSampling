"""Paper-style metrics on fixed queries; local-box/domain and backend controls.

This is a one-model pilot, not a rerun of the seven-dataset paper.
All methods retrieve the same anchor groups and score with the same online
objective. PREMAP leaves belonging to a selected anchor are all considered.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.tree import DecisionTreeRegressor
import torch
import yaml

from certcf import BuildRequest, BuildResult, CertCFAtlas, InputDomain, RegionGeometry
from certcf.certification.premap import PremapBackend, PremapConfig
from experiments.atlas_premap import explicit_graph
from experiments.network_complexity import sha256_file
from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner, _atomic_json


METHODS = ("certcf_l1", "box_crown", "box_premap")


def local_domains(anchors, labels, selected, alpha):
    """Clearances use ALL original anchors, not only the pilot subset."""
    if not 0 < alpha <= 1:
        raise ValueError("box_alpha must be in (0, 1]")
    domains = []
    for idx in selected:
        label = int(labels[idx])
        clearance = cdist(anchors[idx:idx + 1].astype(float), anchors[labels != label].astype(float), "chebyshev").min()
        radius = alpha * clearance
        lo = (anchors[idx].astype(float) - radius).astype(np.float32)
        hi = (anchors[idx].astype(float) + radius).astype(np.float32)
        if np.any(lo >= hi):
            raise ValueError("Degenerate local box")
        box = RegionGeometry(np.empty((0, anchors.shape[1])), [], lo, hi)
        domains.append(InputDomain(str(int(idx)), label, box))
    return tuple(domains)


def query_grouped(regions_by_anchor, anchor_x, anchor_y, selected, query, target, options):
    """Same top-k source anchors, exhaustive leaves within each selected group.

    Empty anchor groups are kept in the ordering. If the first k groups fail,
    inspect the remaining anchors one by one, exactly as the original fallback.
    """
    candidates = selected[anchor_y[selected] == target]
    order = candidates[np.argsort(np.abs(anchor_x[candidates] - query).sum(1), kind="stable")]
    k = int(options["k_anchors"])
    settings = {key: value for key, value in options.items() if key != "k_anchors"}
    batches = [order[:k], *[order[j:j + 1] for j in range(k, len(order))]]
    for batch in batches:
        regions = tuple(r for idx in batch for r in regions_by_anchor.get(str(int(idx)), ()))
        if not regions:
            continue
        atlas = CertCFAtlas.from_regions(BuildResult(regions, (0, 1), (len(query),)), norm=1, **settings)
        cf = atlas.find_counterfactual(query, target, method="sorted", delta=0)
        if cf.success:
            return cf.x_cf
    return None


def nonempty_regions(regions):
    """Remove empty leaves without shrinking, adding anchors, or changing bounds."""
    kept = []
    for r in regions:
        g = r.geometry
        if g.ball is not None:
            raise ValueError("Box feasibility check cannot ignore an Lp ball")
        lp = linprog(np.zeros(g.dimension), A_ub=-g.A, b_ub=g.b,
                     bounds=list(zip(g.lower, g.upper)), method="highs")
        if lp.status == 2:
            continue
        if not lp.success or not g.contains(lp.x, tolerance=1e-6):
            raise RuntimeError("Unresolved region feasibility")
        kept.append(r)
    return tuple(kept)


def predict(model, x):
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(x[i:i + 4096], dtype=torch.float32)).argmax(1).numpy()
                               for i in range(0, len(x), 4096)])


def metric_rows(records, test, train, train_labels, train_predictions, model, cfg):
    """Return per-query metrics and all-success/shared-success summaries.

    Same definitions as DatasetMetrics.ipynb. On numerical-only data the five
    categorical-flip grid entries become five Gaussian-noise replicates. Common
    per-query random numbers are used across methods for a paired comparison.
    """
    frame = pd.DataFrame(records).sort_values(["method", "query_id"])
    lof = LocalOutlierFactor(n_neighbors=cfg["lof_neighbors"], novelty=True, n_jobs=1).fit(train)
    targets = 1 - predict(model, test)
    nn_scale_dist = np.empty(len(test))
    for c in (0, 1):
        neighbors = NearestNeighbors(n_neighbors=1, metric="manhattan", n_jobs=1).fit(train[train_predictions == c])
        nn_scale_dist[targets == c] = neighbors.kneighbors(test[targets == c])[0][:, 0]
    scale = float(nn_scale_dist.mean())
    # Match the paper notebook's same-class RPR reference: training labels.
    ref_nn, ref_distances = {}, []
    for c in (0, 1):
        ref = train[train_labels == c]
        ref_nn[c] = NearestNeighbors(n_neighbors=2, metric="manhattan", n_jobs=1).fit(ref)
        ref_distances.extend(ref_nn[c].kneighbors(ref)[0][:, 1])
    reference_scale = float(np.mean(ref_distances))
    frame["l1"] = np.nan
    frame["l0"] = np.nan
    frame["log10_lof"] = np.nan
    frame["robustness_pct"] = np.nan
    frame["rpr"] = np.nan
    for index, row in frame.iterrows():
        if not row.success:
            continue
        cf64 = np.asarray(row.x_cf, dtype=np.float64)
        cf = cf64.astype(np.float32)
        query_id, target = int(row.query_id), int(row.target)
        delta = np.abs(cf64 - test[query_id].astype(float))
        frame.loc[index, ["l1", "l0"]] = [delta.sum(), (delta > cfg["l0_tolerance"]).sum()]
        frame.loc[index, "log10_lof"] = np.log10(max(float(-lof.score_samples(cf[None])[0]), 1e-12))
        frame.loc[index, "rpr"] = ref_nn[target].kneighbors(cf[None], n_neighbors=1)[0][0, 0] / reference_scale
        fractions = []
        for sigma_index, sigma in enumerate(cfg["sigma_grid"]):
            for repeat in range(cfg["robustness_repeats"]):
                rng = np.random.default_rng(np.random.SeedSequence([cfg["random_state"], query_id, sigma_index, repeat]))
                noisy = cf + rng.normal(0, sigma, (cfg["perturbations_per_cell"], len(cf)))
                fractions.append(float(np.mean(predict(model, noisy) == target)))
        frame.loc[index, "robustness_pct"] = 100 * np.mean(fractions)
    common = set.intersection(*(set(frame.loc[(frame.method == m) & frame.success, "query_id"]) for m in METHODS))
    summaries = []
    for population in ("all_successes", "shared_successes"):
        for method in METHODS:
            all_rows = frame[frame.method == method]
            valid = all_rows[all_rows.success]
            if population == "shared_successes":
                valid = valid[valid.query_id.isin(common)]
            tree_error = None
            if len(valid) >= 20:
                positions = valid.query_id.to_numpy(dtype=int)
                outputs = np.stack(valid.x_cf.map(np.asarray)).astype(float)
                train_idx, test_idx = train_test_split(np.arange(len(valid)), test_size=cfg["tree_test_size"], random_state=cfg["random_state"])
                tree = DecisionTreeRegressor(max_depth=cfg["tree_depth"], min_samples_leaf=cfg["tree_min_samples_leaf"], random_state=cfg["random_state"])
                tree.fit(test[positions[train_idx]].astype(float), outputs[train_idx])
                error = np.abs(tree.predict(test[positions[test_idx]].astype(float)) - outputs[test_idx]).sum(1)
                tree_error = float(error.mean() / scale)
            summary = {"population": population, "method": method, "n_queries": len(all_rows),
                       "n_success": int(all_rows.success.sum()), "validity_pct": float(100 * all_rows.success.mean()),
                       "n_quality": len(valid), "interpretability_proxy": tree_error,
                       "query_seconds_all": float(all_rows.seconds.mean()), "nn_l1_normalizer": scale}
            summary.update({metric: float(valid[metric].mean()) if len(valid) else None
                            for metric in ("l1", "l0", "robustness_pct", "log10_lof", "rpr")})
            summary["query_seconds_quality_subset"] = float(valid.seconds.mean()) if len(valid) else None
            summaries.append(summary)
    return frame, summaries


def load_source(cfg):
    """Read an existing experiment without preparing, training, or rebuilding it."""
    if cfg.get("source_kind", "synthetic32") == "tabular":
        from experiments.lirpa_refinement_ablation import LiRPARefinementAblationRunner
        source = LiRPARefinementAblationRunner.from_yaml(cfg["source_config"])
        case = source.resolve_cases([cfg["dataset"]])[0]
        geometry = source._geometry(case["id"])  # validates saved data/model hashes
        if not source._valid_build(case):
            raise RuntimeError("Existing tabular atlas invalid; refusing to rebuild")
        train, train_labels, test, _, model, spec = source._load_case(case, "cpu")
        if spec.categorical_features or source._fixed_dims(case).size:
            raise ValueError("This metric pilot only supports unconstrained numerical datasets")
        np.testing.assert_array_equal(train[geometry["support_indices"]], geometry["x_support"])
        np.testing.assert_array_equal(test[geometry["query_indices"]], geometry["x_query"])
        atlas_dir = source.paths.atlas(case["id"])
        tracked = [source._checkpoint(case), source._dataset_source(case),
                   source.paths.geometry(case["id"]), *sorted(atlas_dir.glob("*"))]
        hashes = lambda: {str(p): sha256_file(p) for p in tracked if p.is_file()}
        return dict(train=train, train_labels=train_labels, test=geometry["x_query"],
                    anchors=geometry["x_anchor"], anchor_labels=geometry["y_anchor_pred"],
                    model=model, atlas_dir=atlas_dir, margin=source.config["certcf"]["classification_margin"],
                    fingerprint=source._run_fingerprint(case), hashes=hashes)
    if cfg.get("source_kind", "synthetic32") != "synthetic32":
        raise ValueError("Unknown source_kind")
    source = TopKHeuristicAblationRunner.from_yaml(cfg["source_config"])
    if not source._valid_build():
        raise RuntimeError("Existing source atlas invalid; refusing to rebuild or retrain")
    with np.load(source.source_runner.paths.dataset, allow_pickle=False) as data:
        train, test, train_labels = data["x_train"].copy(), data["x_test"].copy(), data["y_train"].copy()
    with np.load(source.paths.atlas_support, allow_pickle=False) as data:
        anchors, anchor_labels = data["x_support"].copy(), data["y_support"].copy()
    model = source.source_runner._load_model(source.depth, source.width, "cpu")
    return dict(train=train, test=test, train_labels=train_labels, anchors=anchors,
                anchor_labels=anchor_labels, model=model, atlas_dir=source.paths.atlas_dir,
                margin=source.config["certcf"]["classification_margin"],
                fingerprint=source._run_fingerprint(), hashes=source._atlas_file_hashes)


def run(config_path, python, repository):
    cfg = yaml.safe_load(Path(config_path).read_text())
    torch.set_num_threads(1)
    source = load_source(cfg)
    source_hashes = source["hashes"]()
    train, test, train_labels = (source[k] for k in ("train", "test", "train_labels"))
    anchors, anchor_labels, model = (source[k] for k in ("anchors", "anchor_labels", "model"))
    graph = explicit_graph(model, torch.tensor(train[:128], dtype=torch.float32))
    predictions, train_predictions = predict(graph, test), predict(graph, train)
    np.testing.assert_array_equal(predict(graph, anchors), anchor_labels)
    rng = np.random.default_rng(cfg["seed"])
    selected = np.concatenate([np.sort(rng.choice(np.flatnonzero(anchor_labels == c), cfg["anchors_per_class"], replace=False)) for c in (0, 1)])
    domains = local_domains(anchors, anchor_labels, selected, cfg["box_alpha"])
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    margin = source["margin"]
    baseline = CertCFAtlas.from_directory(source["atlas_dir"], classification_margin=margin)
    lookup = {tuple(x): int(i) for i, x in enumerate(anchors)}
    base_regions = tuple(replace(r, source_domain_id=str(lookup[tuple(r.representative)]))
                         for r in baseline.regions if lookup[tuple(r.representative)] in selected)
    code = [Path(__file__), Path("src/experiments/atlas_premap.py"), Path("src/certcf/atlas.py"),
            Path("src/certcf/certification/premap.py"), Path("src/certcf/certification/premap_worker.py"),
            Path("src/certcf/geometry/regions.py"), Path("notebooks/DatasetMetrics.ipynb")]
    identity = {"config": cfg, "source_run": source["fingerprint"], "source_atlas": source_hashes,
                "selected_anchor_indices": selected.tolist(), "code": {str(p): sha256_file(p) for p in code}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    meta_path = root / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["fingerprint"] != fingerprint:
        raise RuntimeError("Stale pilot directory; choose a new output_dir")
    _atomic_json({"fingerprint": fingerprint, "identity": identity, "margin": margin,
                  "notes": [f"{len(selected)}-anchor pilot, selected from {len(anchors)} saved anchors", "Same source-anchor retrieval; all leaves per selected anchor are evaluated",
                            "Serial query timings include group selection and subset atlas setup; not comparable to paper parallel timing",
                            "Erosion disabled; robustness measured empirically", "Box endpoints explicitly rounded to float32; no anchor-inclusion or shrinkage"]}, meta_path)
    atlases = {"certcf_l1": BuildResult(base_regions, (0, 1), (test.shape[1],))}
    build_rows = [{"method": "certcf_l1", "n_regions": len(base_regions), "offline_seconds": None,
                   "note": "Reused original certified regions; offline timing not remeasured"}]
    for name in ("box_crown", "box_premap"):
        combined, seconds, returned = [], 0.0, 0
        for label in (0, 1):
            folder = root / name / f"class_{label}"
            completed = folder / "completed.json"
            if completed.exists():
                saved = json.loads(completed.read_text())
                if saved["fingerprint"] != fingerprint or sha256_file(folder / "atlas" / "manifest.json") != saved["manifest_sha256"]:
                    raise RuntimeError("Stale build checkpoint")
                built = CertCFAtlas.from_directory(folder / "atlas").build_result
                elapsed = saved["seconds"]
            else:
                print(f"Building {name}, class={label}, anchors={cfg['anchors_per_class']}...", flush=True)
                options = dict(cfg["premap"])
                options["max_regions"] = 1 if name == "box_crown" else options["max_regions"]
                backend = PremapBackend(PremapConfig(python, repository, seed=cfg["seed"], optimize_bounds=name == "box_premap", **options))
                request = BuildRequest(graph, tuple(d for d in domains if d.target_class == label), (0, 1), (test.shape[1],), margin)
                started = time.perf_counter()
                built = backend.build(request)
                elapsed = time.perf_counter() - started
                CertCFAtlas.from_regions(built).save_bounds(folder / "atlas")
                _atomic_json({"fingerprint": fingerprint, "seconds": elapsed,
                              "manifest_sha256": sha256_file(folder / "atlas" / "manifest.json")}, completed)
            seconds += elapsed
            returned += len(built.regions)
            if len(built.diagnostics.get("domain_stats", [])) != cfg["anchors_per_class"]:
                raise RuntimeError(f"{name}: timeout prevented some anchors from being processed; don't silently compare different anchor sets")
            nonempty = nonempty_regions(built.regions)
            combined.extend(replace(r, region_id=f"{label}:{r.region_id}", representative=anchors[int(r.source_domain_id)]) for r in nonempty)
            print(f"{name} class={label}: {len(nonempty)}/{len(built.regions)} nonempty leaves, {elapsed:.2f}s", flush=True)
        atlases[name] = BuildResult(tuple(combined), (0, 1), (test.shape[1],))
        build_rows.append({"method": name, "n_regions": len(combined), "returned_leaves": returned, "offline_seconds": seconds,
                           "anchors_with_regions": len({r.source_domain_id for r in combined})})
    groups = {}
    for method, result in atlases.items():
        groups[method] = {str(int(i)): tuple(r for r in result.regions if r.source_domain_id == str(int(i))) for i in selected}
        atlas = CertCFAtlas.from_regions(result)
        atlas.save_bounds(root / method / "combined_atlas")
        for label in (0, 1):
            membership = atlas.membership_matrix(test, label)
            if membership[predictions != label].any():
                raise AssertionError("Certificate contradicts held-out predictions")
            build_rows[METHODS.index(method)][f"coverage_class_{label}"] = float(membership[predictions == label].any(1).mean())
    _atomic_json(build_rows, root / "build_summary.json")
    if any(not any(r.target_class == c for r in atlases[m].regions) for m in METHODS for c in (0, 1)):
        raise RuntimeError("A variant has no usable regions for a class; pilot build saved, full query comparison not launched")
    pilot = np.concatenate([rng.permutation(np.flatnonzero(predictions == c))[:cfg["pilot_queries_per_class"]] for c in (0, 1)])
    order = np.concatenate([pilot, np.setdiff1d(np.arange(len(test)), pilot)])
    records = []
    for position, query_id in enumerate(order):
        target = int(1 - predictions[query_id])
        # Rotate method execution order to avoid always giving one variant first.
        for method in np.roll(np.array(METHODS), position % len(METHODS)):
            path = root / method / "queries" / f"{query_id:04d}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["fingerprint"] != fingerprint:
                    raise RuntimeError("Stale query artifact")
            else:
                started = time.perf_counter()
                point = query_grouped(groups[method], anchors, anchor_labels, selected, test[query_id], target, cfg["query"])
                elapsed = time.perf_counter() - started
                success = point is not None
                if success:
                    if int(predict(graph, np.asarray(point)[None])[0]) != target:
                        raise AssertionError("Returned CF is not target-valid")
                    if not CertCFAtlas.from_regions(atlases[method]).membership_matrix(np.asarray(point)[None], target).any():
                        raise AssertionError("CF is outside every certified region")
                record = {"fingerprint": fingerprint, "method": str(method), "query_id": int(query_id),
                          "target": target, "success": success, "seconds": elapsed,
                          "x_cf": np.asarray(point).tolist() if success else None}
                _atomic_json(record, path)
            records.append(record)
        if position + 1 == len(pilot):
            pilot_summary = {m: sum(r["success"] for r in records if r["method"] == m) for m in METHODS}
            _atomic_json({"queries": len(pilot), "successes": pilot_summary}, root / "pilot.json")
            print(f"Pilot ({len(pilot)} queries): {pilot_summary}. Extending same atlases to all {len(test)} queries.", flush=True)
        if (position + 1) % 50 == 0:
            print(f"Completed {position + 1}/{len(test)} queries across all variants", flush=True)
    print("Computing paper-style quality/robustness metrics...", flush=True)
    per_query, summaries = metric_rows(records, test, train, train_labels, train_predictions, graph, cfg["metrics"])
    per_query.drop(columns=["fingerprint"]).to_parquet(root / "metrics_per_query.parquet", index=False)
    pd.DataFrame(summaries).to_csv(root / "metrics_summary.csv", index=False)
    _atomic_json(summaries, root / "metrics_summary.json")
    if source["hashes"]() != source_hashes:
        raise AssertionError("Source atlas changed")
    return {"output_dir": str(root), "builds": build_rows, "metrics": summaries}
