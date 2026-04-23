from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from certcf.atlas import CVXPY_AVAILABLE, CertCFAtlas
from counterfactuals.methods.certcf import CertCF


def _tiny_dataset() -> TensorDataset:
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    return TensorDataset(x, y)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(2, 2))


def _manual_cvxpy_atlas() -> CertCFAtlas:
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.norm = 2
    atlas.distance_norm = 2
    atlas.ohe_slices = None
    atlas.solver_maxiter = 200
    atlas.ohe_decode_mode = "exact"
    atlas.decode_beam_width = 8
    atlas.decode_beam_branch_top_k = 3
    atlas.decode_beam_max_solver_calls = 32
    atlas.cvxpy_solvers = ["CLARABEL", "SCS"]
    atlas.cvxpy_solver_options = {}
    atlas.cvxpy_accept_statuses = {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal", "optimal_inaccurate"],
    }
    return atlas


def test_certcf_defaults_cvxpy_solver_config():
    method = CertCF(model=object())

    assert method.cvxpy_solvers == ["CLARABEL", "SCS"]
    assert method.cvxpy_solver_options == {}
    assert method.cvxpy_accept_statuses == {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal", "optimal_inaccurate"],
    }
    assert method.ohe_decode_mode == "exact"
    assert method.decode_beam_width == 8
    assert method.decode_beam_branch_top_k == 3
    assert method.decode_beam_max_solver_calls == 32


def test_atlas_direct_constructor_defaults_cvxpy_solver_config():
    atlas = CertCFAtlas(_tiny_model(), _tiny_dataset(), device="cpu")

    assert atlas.cvxpy_solvers == ["CLARABEL", "SCS"]
    assert atlas.cvxpy_solver_options == {}
    assert atlas.cvxpy_accept_statuses == {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal", "optimal_inaccurate"],
    }
    assert atlas.ohe_decode_mode == "exact"
    assert atlas.decode_beam_width == 8
    assert atlas.decode_beam_branch_top_k == 3
    assert atlas.decode_beam_max_solver_calls == 32


def test_certcf_normalizes_ohe_decode_config():
    method = CertCF(
        model=object(),
        ohe_decode_mode=" Beam_Then_Exact ",
        decode_beam_width=5,
        decode_beam_branch_top_k=2,
        decode_beam_max_solver_calls=11,
    )

    assert method.ohe_decode_mode == "beam_then_exact"
    assert method.decode_beam_width == 5
    assert method.decode_beam_branch_top_k == 2
    assert method.decode_beam_max_solver_calls == 11


@pytest.mark.parametrize("bad_mode", ["", "greedy", 123])
def test_certcf_rejects_invalid_ohe_decode_mode(bad_mode):
    with pytest.raises(ValueError, match="ohe_decode_mode"):
        CertCF(model=object(), ohe_decode_mode=bad_mode)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("decode_beam_width", 0),
        ("decode_beam_branch_top_k", 0),
        ("decode_beam_max_solver_calls", 0),
    ],
)
def test_certcf_rejects_non_positive_beam_params(field_name, field_value):
    kwargs = {field_name: field_value}
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        CertCF(model=object(), **kwargs)


def test_certcf_normalizes_cvxpy_solver_names():
    method = CertCF(
        model=object(),
        cvxpy_solvers=["clarabel", "scs"],
        cvxpy_solver_options={"clarabel": {"max_iter": 12}},
        cvxpy_accept_statuses={"scs": ["optimal_inaccurate"]},
    )

    assert method.cvxpy_solvers == ["CLARABEL", "SCS"]
    assert method.cvxpy_solver_options == {"CLARABEL": {"max_iter": 12}}
    assert method.cvxpy_accept_statuses == {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal_inaccurate"],
    }


def test_certcf_rejects_duplicate_cvxpy_solver_names():
    with pytest.raises(ValueError, match="cvxpy_solvers must be unique"):
        CertCF(model=object(), cvxpy_solvers=["clarabel", "CLARABEL"])


def test_certcf_rejects_unknown_solver_references_in_nested_config():
    with pytest.raises(ValueError, match="cvxpy_solver_options may only reference configured solvers"):
        CertCF(
            model=object(),
            cvxpy_solvers=["CLARABEL"],
            cvxpy_solver_options={"SCS": {"eps": 1e-3}},
        )

    with pytest.raises(ValueError, match="cvxpy_accept_statuses may only reference configured solvers"):
        CertCF(
            model=object(),
            cvxpy_solvers=["CLARABEL"],
            cvxpy_accept_statuses={"SCS": ["optimal"]},
        )


def test_certcf_rejects_invalid_accept_statuses():
    with pytest.raises(ValueError, match="cvxpy_accept_statuses entries must be non-empty lists of strings"):
        CertCF(
            model=object(),
            cvxpy_solvers=["CLARABEL"],
            cvxpy_accept_statuses={"CLARABEL": []},
        )

    with pytest.raises(ValueError, match="cvxpy_accept_statuses entries must be non-empty lists of strings"):
        CertCF(
            model=object(),
            cvxpy_solvers=["CLARABEL"],
            cvxpy_accept_statuses={"CLARABEL": [123]},
        )


def test_atlas_direct_constructor_normalizes_cvxpy_solver_config():
    atlas = CertCFAtlas(
        _tiny_model(),
        _tiny_dataset(),
        device="cpu",
        cvxpy_solvers=["clarabel", "scs"],
        cvxpy_solver_options={"clarabel": {"max_iter": 17}},
        cvxpy_accept_statuses={"scs": ["optimal_inaccurate"]},
    )

    assert atlas.cvxpy_solvers == ["CLARABEL", "SCS"]
    assert atlas.cvxpy_solver_options == {"CLARABEL": {"max_iter": 17}}
    assert atlas.cvxpy_accept_statuses == {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal_inaccurate"],
    }


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise the configurable solve loop.")
def test_project_cvxpy_respects_solver_order_kwargs_and_accept_statuses(monkeypatch):
    atlas = _manual_cvxpy_atlas()
    atlas.cvxpy_solvers = ["CLARABEL", "SCS"]
    atlas.cvxpy_solver_options = {
        "CLARABEL": {"max_iter": 7},
        "SCS": {"eps": 1e-3, "max_iters": 55},
    }
    atlas.cvxpy_accept_statuses = {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal", "optimal_inaccurate"],
    }

    calls: list[tuple[str, dict[str, Any]]] = []
    statuses = {
        "CLARABEL": "solver_error",
        "SCS": "optimal_inaccurate",
    }

    def fake_solve(self, *, solver, verbose=False, warm_start=False, **kwargs):
        del verbose
        calls.append((solver, {"warm_start": warm_start, **kwargs}))
        if statuses[solver] == "solver_error":
            raise RuntimeError("boom")
        self._status = statuses[solver]
        self.variables()[0].value = np.array([0.0, 0.0], dtype=np.float64)

    monkeypatch.setattr("certcf.atlas.cp.Problem.solve", fake_solve)

    x_proj, dist, profile = atlas._project_cvxpy(
        x0=np.array([0.0, 0.0], dtype=np.float64),
        A_full=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        b_full=np.array([0.0, 0.0], dtype=np.float64),
        center=np.array([0.0, 0.0], dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
    )

    assert x_proj is not None
    assert np.isclose(dist, 0.0)
    assert calls == [
        ("CLARABEL", {"warm_start": True, "max_iter": 7}),
        ("SCS", {"warm_start": True, "eps": 1e-3, "max_iters": 55}),
    ]
    assert profile["cvxpy_solver_used"] == "SCS"
    assert profile["cvxpy_solver_status"] == "optimal_inaccurate"
    assert profile["cvxpy_solver_attempts"] == 2
    assert profile["cvxpy_fallback_used"] is True


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise the configurable solve loop.")
def test_project_cvxpy_failure_after_all_solvers_returns_inf_and_profiles_last_attempt(monkeypatch):
    atlas = _manual_cvxpy_atlas()
    atlas.cvxpy_solvers = ["CLARABEL", "SCS"]
    atlas.cvxpy_solver_options = {}
    atlas.cvxpy_accept_statuses = {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal"],
    }

    def fake_solve(self, *, solver, verbose=False, warm_start=False, **kwargs):
        del verbose, warm_start, kwargs
        self._status = "infeasible"

    monkeypatch.setattr("certcf.atlas.cp.Problem.solve", fake_solve)

    x_proj, dist, profile = atlas._project_cvxpy(
        x0=np.array([0.0, 0.0], dtype=np.float64),
        A_full=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        b_full=np.array([0.0, 0.0], dtype=np.float64),
        center=np.array([0.0, 0.0], dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
    )

    assert x_proj is None
    assert np.isinf(dist)
    assert profile["cvxpy_solver_used"] == "SCS"
    assert profile["cvxpy_solver_status"] == "infeasible"
    assert profile["cvxpy_solver_attempts"] == 2
    assert profile["cvxpy_fallback_used"] is True


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise projection profiling.")
def test_project_onto_polytope_merges_cvxpy_solver_profile(monkeypatch):
    atlas = _manual_cvxpy_atlas()

    def fake_project_cvxpy(*args, **kwargs):
        del args, kwargs
        return (
            np.array([0.0, 0.0], dtype=np.float64),
            0.0,
            {
                "cvxpy_solver_used": "CLARABEL",
                "cvxpy_solver_status": "optimal",
                "cvxpy_solver_attempts": 1,
                "cvxpy_fallback_used": False,
            },
        )

    monkeypatch.setattr(atlas, "_project_cvxpy", fake_project_cvxpy)

    x_proj, dist, profile = atlas._project_onto_polytope(
        x0=np.array([0.0, 0.0], dtype=np.float64),
        A=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        b=np.array([0.0, 0.0], dtype=np.float64),
        center=np.array([0.0, 0.0], dtype=np.float64),
        eps_i=1.0,
    )

    assert x_proj is not None
    assert np.isclose(dist, 0.0)
    assert profile["cvxpy_solver_used"] == "CLARABEL"
    assert profile["cvxpy_solver_status"] == "optimal"
    assert profile["cvxpy_solver_attempts"] == 1
    assert profile["cvxpy_fallback_used"] is False


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise projection profiling.")
def test_solver_maxiter_remains_slsqp_only(monkeypatch):
    atlas = _manual_cvxpy_atlas()
    atlas.norm = np.inf
    atlas.distance_norm = 2

    recorded = {}

    def fake_slsqp(
        x0,
        A_full,
        b_full,
        center,
        box_eps,
        maxiter,
        tol,
        fixed_dims=None,
        ohe_slices=None,
        fixed_ohe_assignments=None,
    ):
        del x0, A_full, b_full, center, box_eps, tol, fixed_dims, ohe_slices, fixed_ohe_assignments
        recorded["maxiter"] = maxiter
        return np.array([0.0, 0.0], dtype=np.float64), 0.0

    monkeypatch.setattr(atlas, "_project_slsqp", fake_slsqp)

    x_proj, dist = atlas._solve_projection_subproblem(
        x0=np.array([0.0, 0.0], dtype=np.float64),
        A_full=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        b_full=np.array([0.0, 0.0], dtype=np.float64),
        center=np.array([0.0, 0.0], dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=123,
        tol=1e-9,
    )

    assert x_proj is not None
    assert np.isclose(dist, 0.0)
    assert recorded["maxiter"] == 123
