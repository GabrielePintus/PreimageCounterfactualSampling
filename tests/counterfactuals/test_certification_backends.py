import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from certcf import CertCFAtlas, CertifiedRegion, RegionGeometry, NormBall, BuildRequest, BuildResult, InputDomain
from certcf.certification.artifacts import load_regions, save_regions
from certcf.certification.local_lirpa import LocalLiRPABackend
from certcf.certification.premap import PremapBackend, PremapConfig, export_sequential
from certcf.certification import premap as premap_module


def geometry(ball=None):
    return RegionGeometry(np.array([[1., 0.]]), np.array([-0.4]), np.array([-1., -2.]), np.array([1., 2.]), ball)


def result():
    # Representative deliberately outside the region: never accepted as a CF.
    r = CertifiedRegion("a", 1, geometry(), "root", np.array([-0.9, -1.9]), {"test_fixture": True})
    return BuildResult((r,), (0, 1), (2,))


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_geometry_erosion(p):
    g = geometry(NormBall(np.zeros(2), 0.9, p))
    e = g.eroded(.1, p)
    assert e.ball.radius == pytest.approx(.8)
    np.testing.assert_allclose(e.lower, [-.9, -1.9])
    assert not e.contains([.45, 0])
    assert e.contains([.6, 0])
    assert g.eroded(3, p) is None
    with pytest.raises(ValueError):
        g.eroded(-1, p)


def test_geometry_validation():
    with pytest.raises(ValueError):
        RegionGeometry([[np.nan, 0]], [0], [-1, -1], [1, 1])
    with pytest.raises(ValueError):
        RegionGeometry([[1, 0]], [0], [1, -1], [-1, 1])
    assert not geometry().contains([np.nan, 0])


@pytest.mark.parametrize("method", ["sorted", "nearest_anchor"])
@pytest.mark.parametrize("delta", [0., .1])
def test_generic_projection_preserves_box_and_margin(method, delta):
    atlas = CertCFAtlas.from_regions(result(), distance_norm=1, classification_margin=.3)
    cf = atlas.find_counterfactual(np.array([-1., 0.]), 1, method=method, delta=delta, robust_norm=1)
    assert cf.success
    # No second application of the atlas classification margin.
    assert cf.x_cf[0] == pytest.approx(.4 + delta, abs=1e-5)
    assert abs(cf.x_cf[1]) < 1e-5
    assert atlas.regions[0].geometry.eroded(delta, 1).contains(cf.x_cf)
    assert not atlas.find_counterfactual(np.array([-1., 0.]), 0, method=method).success


def test_generic_fixed_feature_and_degenerate_geometry():
    atlas = CertCFAtlas.from_regions(result(), distance_norm=1)
    assert not atlas.find_counterfactual(np.array([-1., 0.]), 1, fixed_dims=np.array([0])).success
    r = CertifiedRegion("p", 1, RegionGeometry(np.empty((0, 2)), np.empty(0), [0, 0], [0, 0]), "point")
    atlas = CertCFAtlas.from_regions(BuildResult((r,), (0, 1), (2,)), distance_norm=1)
    cf = atlas.find_counterfactual(np.ones(2), 1)
    assert cf.success
    np.testing.assert_allclose(cf.x_cf, 0, atol=1e-6)


def test_artifact_roundtrip_and_corruption(tmp_path):
    atlas = CertCFAtlas.from_regions(result(), distance_norm=1, classification_margin=.12)
    atlas.save_bounds(tmp_path)
    loaded = CertCFAtlas.from_directory(tmp_path)
    assert loaded.model is None and loaded._preimage is None
    assert loaded.classification_margin == .12
    assert loaded.distance_norm == 1
    cf = loaded.find_counterfactual(np.array([-1., 0.]), 1)
    assert cf.success
    with pytest.raises(RuntimeError, match="requires a classifier"):
        loaded.verify_counterfactual(cf.x_cf, 1)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    with (tmp_path / manifest["file"]).open("ab") as handle:
        handle.write(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        load_regions(tmp_path)


@pytest.mark.parametrize("version", [1, 2])
def test_solver_and_execution_settings_roundtrip(tmp_path, version):
    options = dict(solver_maxiter=123, cvxpy_solvers=["CLARABEL"],
                   cvxpy_solver_options={"CLARABEL": {"max_iter": 77, "tol_gap_abs": 1e-8}},
                   cvxpy_accept_statuses={"CLARABEL": ["optimal"]},
                   query_parallelism=1, candidate_parallelism=2,
                   candidate_parallel_backend="process", classification_margin=.01, distance_norm=1)
    atlas = CertCFAtlas.from_regions(result(), **options)
    if version == 1:
        # Minimal legacy numeric payload: persistence must support both formats.
        atlas.bounds = {label: {"X": np.zeros((1, 2)), "eps": np.ones(1),
                               "lA": np.array([[[1., 0.]]]), "lbias": np.array([[-.4]])}
                        for label in atlas.class_labels}
    atlas.save_bounds(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["format_version"] == version
    loaded = CertCFAtlas.from_directory(tmp_path)
    assert loaded._saved_query_settings() == atlas._saved_query_settings()
    override = CertCFAtlas.from_directory(tmp_path, solver_maxiter=456,
                                          cvxpy_solver_options={"CLARABEL": {"max_iter": 99}})
    assert override.solver_maxiter == 456
    assert override.cvxpy_solver_options == {"CLARABEL": {"max_iter": 99}}
    assert override.candidate_parallelism == 2
    # Loading only geometry intentionally preserves an existing query policy.
    loaded.solver_maxiter, loaded.norm, loaded.classification_margin = 42, 1, .8
    loaded.load_bounds(tmp_path)
    assert loaded.solver_maxiter == 42
    assert loaded.norm == atlas.norm and loaded.classification_margin == .01


@pytest.mark.parametrize("version", [1, 2])
def test_old_artifacts_without_solver_settings_use_defaults(tmp_path, version):
    if version == 2:
        save_regions(result(), tmp_path, settings={"distance_norm": 1})
    else:
        np.savez(tmp_path / "class_0.npz", X=np.zeros((1, 2)), eps=np.ones(1))
        (tmp_path / "manifest.json").write_text(json.dumps({
            "format_version": 1, "class_labels": [0], "model_input_shape": [2],
            "norm": 1, "distance_norm": 1, "files": {"0": "class_0.npz"}}))
    loaded = CertCFAtlas.from_directory(tmp_path)
    assert loaded.solver_maxiter == 500
    assert loaded.cvxpy_solvers == ["CLARABEL", "SCS"]
    assert loaded.candidate_parallelism == 1


def test_saved_settings_are_detached_from_live_options():
    atlas = CertCFAtlas(cvxpy_solver_options={"CLARABEL": {"max_iter": 77}})
    saved = atlas._saved_query_settings()
    saved["cvxpy_solver_options"]["CLARABEL"]["max_iter"] = 1
    assert atlas.cvxpy_solver_options["CLARABEL"]["max_iter"] == 77


def test_premap_sampling_configuration():
    assert PremapConfig("python", "repo").coverage_samples == 2000
    legacy = PremapConfig("python", "repo", samples=17, seed=42)
    explicit = PremapConfig("python", "repo", coverage_samples=17, seed=42)
    assert legacy == explicit
    settings = explicit.sampling_settings()
    assert settings["coverage_samples_per_half"] == 8
    assert settings["optimization_samples"] == 1000
    assert settings["optimization_seed"] == 1
    with pytest.raises(ValueError, match="disagree"):
        PremapConfig("python", "repo", samples=17, coverage_samples=18)
    for count in (True, 1, 2.5):
        with pytest.raises(ValueError, match="coverage_samples"):
            PremapConfig("python", "repo", coverage_samples=count)


def test_empty_partial_artifact(tmp_path):
    partial = BuildResult((), (0, 1), (2,), "budget_exhausted", {"timed_out": True})
    save_regions(partial, tmp_path)
    atlas = CertCFAtlas.from_directory(tmp_path)
    assert atlas.build_result.status == "budget_exhausted"
    assert not atlas.find_counterfactual(np.zeros(2), 1).success
    assert "Total polytopes: 0" in atlas.summary()
    assert atlas.membership_matrix(np.zeros((3, 2)), 1).shape == (3, 0)


def test_generic_membership_columns_and_box_faces():
    atlas = CertCFAtlas.from_regions(result())
    points = np.array([[.5, 0], [.3, 0], [2., 0]])
    np.testing.assert_array_equal(atlas.membership_matrix(points, 1, batch_size=1), [[True], [False], [False]])
    assert not atlas.membership_matrix(points, 0).any()


@pytest.mark.parametrize("parallel_backend", ["thread", "process"])
def test_generic_parallel_projection(parallel_backend):
    from dataclasses import replace
    r = result().regions[0]
    built = BuildResult((r, replace(r, region_id="b")), (0, 1), (2,))
    atlas = CertCFAtlas.from_regions(built, distance_norm=1, candidate_parallelism=2,
                                    candidate_parallel_backend=parallel_backend)
    try:
        cf = atlas.find_counterfactual(np.array([-1., 0.]), 1, method="nearest_anchor", query_k_candidates=2)
        assert cf.success
        assert cf.x_cf[0] == pytest.approx(.4, abs=1e-5)
    finally:
        atlas.close_candidate_process_pool()


def test_generic_ohe_decoding_and_robustness():
    region = CertifiedRegion("cat", 1, RegionGeometry([[0, 1]], [-.2], [0, 0], [1, 1]), "root")
    atlas = CertCFAtlas.from_regions(BuildResult((region,), (0, 1), (2,)), distance_norm=1,
                                    ohe_slices=[(0, 2)], ohe_decode_mode="exact")
    cf = atlas.find_counterfactual(np.array([1., 0.]), 1)
    assert cf.success
    np.testing.assert_allclose(cf.x_cf, [0, 1], atol=1e-6)
    # Eroding the one-hot box excludes its vertices: don't undo erosion on decode.
    assert not atlas.find_counterfactual(np.array([1., 0.]), 1, delta=.1).success


def test_model_free_import_does_not_load_lirpa():
    subprocess.run([sys.executable, "-c", "from certcf import CertCFAtlas; import sys; assert 'auto_LiRPA' not in sys.modules; CertCFAtlas()"], check=True)


def test_legacy_and_strategy_equivalence(tmp_path):
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
        model[2].weight.copy_(torch.tensor([[-1., 1.], [1., -1.]]))
        model[0].bias.zero_()
        model[2].bias.zero_()
    dataset = torch.utils.data.TensorDataset(torch.tensor([[-1., 0.], [1., 0.]]), torch.tensor([0, 1]))
    atlas = CertCFAtlas(model, dataset, "cpu", norm=1, classification_margin=.01, distance_norm=1)
    atlas.build(verbose=False)
    domains = tuple(InputDomain(f"input-{i}", r.target_class, RegionGeometry(
        np.empty((0, 2)), [], r.geometry.lower, r.geometry.upper, r.geometry.ball,
    )) for i, r in enumerate(atlas.regions))
    fresh = LocalLiRPABackend(adaptive_eps=True).build(BuildRequest(model, domains, (0, 1), (2,), .01))
    assert {r.source_domain_id for r in fresh.regions} == {d.domain_id for d in domains}
    converted = CertCFAtlas.from_regions(atlas.build_result, norm=1, distance_norm=1)
    for label in atlas.class_labels:
        for x in (np.zeros(2), np.array([2., 2.])):
            a = atlas.find_counterfactual(x, label)
            b = converted.find_counterfactual(x, label)
            assert a.success == b.success
            assert a.distance == pytest.approx(b.distance, abs=2e-5)
    atlas.save_bounds(tmp_path)
    restored = CertCFAtlas.from_directory(tmp_path)
    assert restored.model is None and restored.classification_margin == .01
    assert restored.find_counterfactual(np.zeros(2), 0).distance == pytest.approx(atlas.find_counterfactual(np.zeros(2), 0).distance)
    restored.norm, restored.classification_margin = 2, .8
    restored.load_bounds(tmp_path)
    assert restored.norm == 1 and restored.classification_margin == .01


def test_backend_injection():
    class Backend:
        def build(self, request):
            assert request.class_labels == (0, 1)
            return result()
    request = BuildRequest(None, (), (0, 1), (2,))
    atlas = CertCFAtlas(certification_backend=Backend(), distance_norm=1).build(request=request)
    assert atlas.find_counterfactual(np.zeros(2), 1).success


def test_sequential_export_is_explicit_and_numeric():
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    layers, arrays = export_sequential(model, 2, 2)
    assert len(layers) == 3
    assert all(a.dtype == np.float32 for a in arrays.values())
    with pytest.raises(TypeError):
        export_sequential(nn.Linear(2, 2), 2, 2)
    with pytest.raises(ValueError):
        export_sequential(model.double(), 2, 2)


def test_premap_rejects_nonbox_domain_without_launching():
    backend = PremapBackend(PremapConfig("missing-python", "missing-repo"))
    ball = NormBall(np.zeros(2), 1, 1)
    domain = InputDomain("a", 1, geometry(ball))
    request = BuildRequest(nn.Sequential(nn.Linear(2, 2)), (domain,), (0, 1), (2,), .001)
    with pytest.raises(ValueError, match="boxes"):
        backend.build(request)


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_premap_hard_timeout_recovers_only_atomic_checkpoints(monkeypatch, tmp_path, with_checkpoint):
    from pathlib import Path
    domain = InputDomain("box", 1, RegionGeometry(np.empty((0, 2)), [], [-1, -1], [1, 1]))
    request = BuildRequest(nn.Sequential(nn.Linear(2, 2)), (domain,), (0, 1), (2,), .01)
    killed = []
    monkeypatch.setattr(premap_module.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=premap_module.PREMAP_REVISION if cmd[-1] == "HEAD" else ""))
    monkeypatch.setattr(premap_module.os, "killpg", lambda pid, sig: killed.append(pid))

    class FakeProcess:
        pid = 987654
        returncode = None

        def __init__(self, cmd, **kwargs):
            self.waits = 0
            root = Path(cmd[-1])
            if with_checkpoint:
                payload = json.loads((root / "request.json").read_text())
                np.savez(root / "ready.npz", r0_A=[[1, 0]], r0_b=[-.1], r0_lower=[-1, -1], r0_upper=[1, 1])
                manifest = {"file": "ready.npz", "count": 1, "diagnostics": {"complete": False},
                            "request_sha256": payload["request_sha256"],
                            "sha256": hashlib.sha256((root / "ready.npz").read_bytes()).hexdigest()}
                (root / "domain0.json").write_text(json.dumps(manifest))
            # An incomplete payload without a manifest must never be imported.
            (root / "unfinished.npz").write_bytes(b"partial")

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fake-worker", timeout)
            self.returncode = -9
            return -9

    monkeypatch.setattr(premap_module.subprocess, "Popen", FakeProcess)
    built = PremapBackend(PremapConfig("fake-python", str(tmp_path), timeout_s=.1)).build(request)
    assert built.status == "budget_exhausted" and built.diagnostics["timed_out"]
    assert len(built.regions) == int(with_checkpoint)
    assert killed == [987654]


@pytest.mark.skipif(not os.environ.get("CERTCF_PREMAP_PYTHON"), reason="External PREMAP environment not configured")
@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("optimize", [False, True])
def test_real_premap_worker(nonlinear, optimize):
    model = (nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2)) if nonlinear
             else nn.Sequential(nn.Linear(2, 2))).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]) if nonlinear else torch.tensor([[-1., 0.], [1., 0.]]))
        if nonlinear:
            model[2].weight.copy_(torch.tensor([[-1., 1.], [1., -1.]]))
            model[2].bias.zero_()
        model[0].bias.zero_()
    domain = InputDomain("box", 1, RegionGeometry(np.empty((0, 2)), np.empty(0), [-1, -1], [1, 1]))
    other = InputDomain("other-class", 0, domain.geometry)
    request = BuildRequest(model, (domain, other), (0, 1), (2,), .01)
    backend = PremapBackend(PremapConfig(os.environ["CERTCF_PREMAP_PYTHON"], os.environ["CERTCF_PREMAP_REPOSITORY"],
                                        timeout_s=60, samples=100, max_regions=4 if optimize else 1,
                                        optimize_bounds=optimize))
    built = backend.build(request)
    assert built.regions, built.diagnostics
    atlas = CertCFAtlas.from_regions(built, distance_norm=1)
    cf = atlas.find_counterfactual(np.array([-1., 0.]), 1)
    # Plain CROWN can have an empty under-approximation even for this simple
    # redundant ReLU graph; the control must not silently optimize or relax it.
    if nonlinear and not optimize:
        assert not cf.success
    else:
        assert cf.success
    if nonlinear and optimize:
        assert len(built.regions) > 1  # exercise actual upstream splitting
        assert len(built.regions) <= 8  # four leaves per source domain
        assert any(np.any(r.geometry.upper - r.geometry.lower < 2) for r in built.regions)
    elif not nonlinear:
        assert cf.x_cf[0] == pytest.approx(.005, abs=2e-5)
    if cf.success:
        assert 2 * cf.x_cf[0] >= .01 - 1e-6
    grid = np.array([[x, 0.] for x in np.linspace(-1, 1, 100)])
    for region in built.regions:
        included = grid[region.geometry.contains(grid, tolerance=0)]
        sign = 1 if region.target_class == 1 else -1
        assert (sign * 2 * included[:, 0] >= .01 - 1e-6).all()
    assert {r.target_class for r in built.regions} == {0, 1}
