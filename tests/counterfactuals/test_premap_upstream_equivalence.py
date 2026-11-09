"""Differential tests against the pinned PREMAP implementation, not PREMAP2."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from torch import nn

from certcf import BuildRequest, CertCFAtlas, InputDomain, RegionGeometry
from certcf.certification.premap import PremapBackend, PremapConfig, export_sequential


pytestmark = pytest.mark.skipif(
    not os.environ.get("CERTCF_PREMAP_PYTHON"), reason="External PREMAP environment not configured")


def nonlinear_model():
    # Two nontrivial ReLU layers and a genuinely piecewise-linear boundary.
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    parameters = [
        ([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], [.15, -.1, .05, -.2]),
        ([[1., -.4, .6, -.2], [-.3, .8, -.1, .5], [.4, .1, -.7, .8]], [-.2, .1, -.15]),
        ([[-.8, .9, -.3], [.7, -.6, .5]], [.05, -.05]),
    ]
    with torch.no_grad():
        for layer, (weight, bias) in zip((model[0], model[2], model[4]), parameters):
            layer.weight.copy_(torch.tensor(weight))
            layer.bias.copy_(torch.tensor(bias))
    return model


def reference(tmp_path, model, *, mode, seed=7, target=1, samples=128, max_regions=4, threshold=.99):
    layers, arrays = export_sequential(model, 2, 2)
    np.savez(tmp_path / "reference.npz", **arrays)
    (tmp_path / "reference.json").write_text(json.dumps(dict(
        layers=layers, mode=mode, seed=seed, target=target, samples=samples,
        max_regions=max_regions, threshold=threshold, margin=.01,
        lower=[-1., -1.], upper=[1., 1.])))
    worker = Path(__file__).resolve().parents[2] / "src/certcf/certification/premap_worker.py"
    oracle = Path(__file__).with_name("helpers") / "premap_reference.py"
    proc = subprocess.run([os.environ["CERTCF_PREMAP_PYTHON"], "-I", str(oracle),
                           os.environ["CERTCF_PREMAP_REPOSITORY"], str(tmp_path), str(worker)],
                          capture_output=True, text=True, timeout=90, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-8000:] + proc.stderr
    return json.loads((tmp_path / "reference-output.json").read_text())


@pytest.mark.parametrize("target,samples", [(0, 2), (1, 17), (1, 128)])
def test_coverage_matches_original_functions_and_optimizer_contract(tmp_path, target, samples):
    data = reference(tmp_path, nonlinear_model(), mode="sampling", target=target, samples=samples)
    assert data["sampling_equivalent"]
    assert data["optimizer_draws"] == [[1000], [1000]]


@pytest.mark.parametrize("target,max_regions,threshold", [(0, 1, .99), (1, 1, .99), (0, 4, .99), (1, 4, .99), (1, 4, .1)])
def test_nonlinear_bounds_splits_and_stopping_match_upstream(tmp_path, target, max_regions, threshold):
    model = nonlinear_model()
    expected = reference(tmp_path, model, mode="build", target=target,
                         max_regions=max_regions, threshold=threshold)
    domain = InputDomain("test-box", target, RegionGeometry(np.empty((0, 2)), [], [-1, -1], [1, 1]))
    config = PremapConfig(os.environ["CERTCF_PREMAP_PYTHON"], os.environ["CERTCF_PREMAP_REPOSITORY"],
                          coverage_samples=128, seed=7, max_regions=max_regions,
                          coverage_threshold=threshold, timeout_s=90)
    built = PremapBackend(config).build(BuildRequest(model, (domain,), (0, 1), (2,), .01))
    # Fixed fixture expectations ensure both stopping paths are really covered,
    # even if the adapter and oracle were accidentally changed in the same way.
    assert expected["complete"] == (threshold == .1)
    assert len(expected["regions"]) == (1 if threshold == .1 else max_regions)
    assert (built.status == "complete") == expected["complete"]
    assert len(built.regions) == len(expected["regions"])
    assert built.diagnostics["domain_stats"][0]["estimated_coverage"] == pytest.approx(expected["coverage"], abs=1e-7)
    assert built.diagnostics["sampling"] == config.sampling_settings()
    assert built.diagnostics["domain_stats"][0]["coverage_seed"] == 7
    CertCFAtlas.from_regions(built).save_bounds(tmp_path / "atlas")
    restored = CertCFAtlas.from_directory(tmp_path / "atlas")
    assert restored.build_result.diagnostics["sampling"] == config.sampling_settings()
    def key(region):
        return tuple(region["lower"] + region["upper"])
    actual = [dict(A=r.geometry.A, b=r.geometry.b,
                   lower=r.geometry.lower.tolist(), upper=r.geometry.upper.tolist()) for r in built.regions]
    for got, want in zip(sorted(actual, key=key), sorted(expected["regions"], key=key)):
        for field in ("lower", "upper", "A", "b"):
            np.testing.assert_allclose(got[field], want[field], rtol=1e-5, atol=2e-6, err_msg=field)
    # Independent geometry/soundness sanity checks over a 2D grid.
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 61), np.linspace(-1, 1, 61)), axis=-1).reshape(-1, 2)
    logits = model(torch.tensor(grid, dtype=torch.float32)).detach().numpy()
    for region in built.regions:
        g = region.geometry
        in_box = ((grid >= g.lower) & (grid <= g.upper)).all(1)
        affine = grid @ g.A.T + g.b
        margin = logits[:, target] - logits[:, 1 - target] - .01
        assert (affine[in_box, 0] <= margin[in_box] + 2e-6).all()
        inside = g.contains(grid, tolerance=0)
        assert (margin[inside] >= -2e-6).all()
