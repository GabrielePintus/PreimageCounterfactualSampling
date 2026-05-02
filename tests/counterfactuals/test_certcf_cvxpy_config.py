from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from certcf.atlas import CVXPY_AVAILABLE, CertCFAtlas
from certcf.eps_strategies import ConstantEpsStrategy
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
    assert method.classification_margin == 0.0
    assert method.fixed_dims is None
    assert method.immutable_features == ()
    assert method.nondecreasing_dims is None
    assert method.nonincreasing_dims is None
    assert method.nondecreasing_features == ()
    assert method.nonincreasing_features == ()
    assert method.adaptive_eps is False
    assert method.adaptive_eps_shrink_factor == 0.5
    assert method.adaptive_eps_max_shrinks == 8
    assert method.adaptive_eps_min == 1.0e-6
    assert method.adaptive_eps_center_tol == 1.0e-6
    assert method.adaptive_eps_binary_search_steps == 0


def test_certcf_normalizes_fixed_dims():
    method = CertCF(
        model=object(),
        fixed_dims=[3, 1, 3, 2],
        immutable_features=["sex", "race"],
        nondecreasing_dims=[5, 4, 5],
        nonincreasing_dims=[6],
        nondecreasing_features=["age"],
        nonincreasing_features=["income"],
    )

    assert np.array_equal(method.fixed_dims, np.array([1, 2, 3], dtype=np.int64))
    assert method.immutable_features == ("sex", "race")
    assert np.array_equal(method.nondecreasing_dims, np.array([4, 5], dtype=np.int64))
    assert np.array_equal(method.nonincreasing_dims, np.array([6], dtype=np.int64))
    assert method.nondecreasing_features == ("age",)
    assert method.nonincreasing_features == ("income",)


@pytest.mark.parametrize("bad_dims", [[1, -1], [[1, 2]], [1.25]])
def test_certcf_rejects_invalid_fixed_dims(bad_dims):
    with pytest.raises(ValueError, match="fixed_dims"):
        CertCF(model=object(), fixed_dims=bad_dims)


def test_certcf_generate_batch_passes_fixed_dims_to_atlas():
    method = CertCF(
        model=object(),
        fixed_dims=[2, 0, 2],
        immutable_features=["sex"],
        nondecreasing_dims=[1],
        nonincreasing_dims=[2],
        nondecreasing_features=["age"],
        nonincreasing_features=["priors_count"],
    )
    method._is_fitted = True
    calls: list[dict[str, Any]] = []

    class FakeAtlas:
        def find_counterfactual_batch(self, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(
                    x_cf=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    success=True,
                    distance=1.5,
                    profiling={"method": "nearest_anchor"},
                )
            ]

    method.atlas = FakeAtlas()
    results = method.generate_batch(
        x=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        target_class=np.array([1], dtype=np.int64),
    )

    assert len(results) == 1
    assert np.array_equal(calls[0]["fixed_dims"], np.array([0, 2], dtype=np.int64))
    assert np.array_equal(calls[0]["nondecreasing_dims"], np.array([1], dtype=np.int64))
    assert np.array_equal(calls[0]["nonincreasing_dims"], np.array([2], dtype=np.int64))
    assert results[0].metadata["fixed_dims_count"] == 2
    assert results[0].metadata["immutable_features"] == "sex"
    assert results[0].metadata["nondecreasing_dims_count"] == 1
    assert results[0].metadata["nonincreasing_dims_count"] == 1
    assert results[0].metadata["nondecreasing_features"] == "age"
    assert results[0].metadata["nonincreasing_features"] == "priors_count"


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
    assert atlas.classification_margin == 0.0
    assert atlas.adaptive_eps is False
    assert atlas.adaptive_eps_shrink_factor == 0.5
    assert atlas.adaptive_eps_max_shrinks == 8
    assert atlas.adaptive_eps_min == 1.0e-6
    assert atlas.adaptive_eps_center_tol == 1.0e-6
    assert atlas.adaptive_eps_binary_search_steps == 0


def test_certcf_rejects_negative_classification_margin():
    with pytest.raises(ValueError, match="classification_margin"):
        CertCF(model=object(), classification_margin=-1e-3)

    with pytest.raises(ValueError, match="classification_margin"):
        CertCFAtlas(
            _tiny_model(),
            _tiny_dataset(),
            device="cpu",
            classification_margin=-1e-3,
        )


def test_certcf_rejects_invalid_adaptive_eps_config():
    with pytest.raises(ValueError, match="adaptive_eps_shrink_factor"):
        CertCF(model=object(), adaptive_eps_shrink_factor=1.0)

    with pytest.raises(ValueError, match="adaptive_eps_max_shrinks"):
        CertCF(model=object(), adaptive_eps_max_shrinks=-1)

    with pytest.raises(ValueError, match="adaptive_eps_min"):
        CertCF(model=object(), adaptive_eps_min=-1e-6)

    with pytest.raises(ValueError, match="adaptive_eps_center_tol"):
        CertCF(model=object(), adaptive_eps_center_tol=-1e-6)

    with pytest.raises(ValueError, match="adaptive_eps_binary_search_steps"):
        CertCF(model=object(), adaptive_eps_binary_search_steps=-1)


def test_atlas_build_forwards_adaptive_eps_config(monkeypatch):
    atlas = CertCFAtlas(
        _tiny_model(),
        _tiny_dataset(),
        device="cpu",
        eps_strategy=ConstantEpsStrategy(0.2),
        classification_margin=0.05,
        adaptive_eps=True,
        adaptive_eps_shrink_factor=0.25,
        adaptive_eps_max_shrinks=3,
        adaptive_eps_min=1.0e-5,
        adaptive_eps_center_tol=1.0e-4,
        adaptive_eps_binary_search_steps=2,
    )
    calls = []

    def fake_compute_all_bounds(**kwargs):
        calls.append(kwargs)
        return {
            0: {
                "lA": np.zeros((1, 1, 2), dtype=np.float32),
                "lbias": np.ones((1, 1), dtype=np.float32),
                "uA": np.zeros((1, 1, 2), dtype=np.float32),
                "ubias": np.ones((1, 1), dtype=np.float32),
                "X": np.array([[0.0, 0.0]], dtype=np.float32),
                "eps": np.array([0.2], dtype=np.float64),
            },
            1: {
                "lA": np.zeros((1, 1, 2), dtype=np.float32),
                "lbias": np.ones((1, 1), dtype=np.float32),
                "uA": np.zeros((1, 1, 2), dtype=np.float32),
                "ubias": np.ones((1, 1), dtype=np.float32),
                "X": np.array([[1.0, 1.0]], dtype=np.float32),
                "eps": np.array([0.2], dtype=np.float64),
            },
        }

    monkeypatch.setattr(atlas._preimage, "compute_all_bounds", fake_compute_all_bounds)
    atlas.build(verbose=False)

    assert calls
    assert calls[0]["classification_margin"] == 0.05
    assert calls[0]["adaptive_eps"] is True
    assert calls[0]["adaptive_eps_shrink_factor"] == 0.25
    assert calls[0]["adaptive_eps_max_shrinks"] == 3
    assert calls[0]["adaptive_eps_min"] == 1.0e-5
    assert calls[0]["adaptive_eps_center_tol"] == 1.0e-4
    assert calls[0]["adaptive_eps_binary_search_steps"] == 2


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


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise the configurable solve loop.")
def test_project_cvxpy_accepts_small_postsolve_ball_residual(monkeypatch):
    atlas = _manual_cvxpy_atlas()
    atlas.norm = 1
    atlas.distance_norm = 1

    def fake_solve(self, *, solver, verbose=False, warm_start=False, **kwargs):
        del solver, verbose, warm_start, kwargs
        self._status = "optimal"
        self.variables()[0].value = np.array([1.0 + 5e-7, 0.0], dtype=np.float64)

    monkeypatch.setattr("certcf.atlas.cp.Problem.solve", fake_solve)

    x_proj, dist, profile = atlas._project_cvxpy(
        x0=np.array([0.0, 0.0], dtype=np.float64),
        A_full=np.array([[0.0, 0.0]], dtype=np.float64),
        b_full=np.array([1.0], dtype=np.float64),
        center=np.array([0.0, 0.0], dtype=np.float64),
        box_eps=2.0,
        ball_eps=1.0,
    )

    assert x_proj is not None
    assert np.isclose(dist, 1.0 + 5e-7)
    assert profile["cvxpy_solver_status"] == "optimal"
    assert profile["cvxpy_fallback_used"] is False


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise directional projection constraints.")
def test_project_cvxpy_rejects_directionally_infeasible_polytope():
    atlas = _manual_cvxpy_atlas()

    x_proj, dist, _ = atlas._project_cvxpy(
        x0=np.array([0.0, 0.0], dtype=np.float64),
        A_full=np.zeros((1, 2), dtype=np.float64),
        b_full=np.ones(1, dtype=np.float64),
        center=np.array([-1.0, 0.0], dtype=np.float64),
        box_eps=0.25,
        ball_eps=0.25,
        nondecreasing_dims=np.array([0], dtype=np.int64),
    )

    assert x_proj is None
    assert np.isinf(dist)


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


def test_erode_constraints_applies_classification_margin_to_lirpa_rows_only():
    atlas = _manual_cvxpy_atlas()
    atlas.classification_margin = 0.25

    A_full, b_full, box_eps, ball_eps = atlas._erode_constraints(
        A=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        b=np.array([0.5, 0.75], dtype=np.float64),
        center=np.array([0.0, 0.0], dtype=np.float64),
        d=2,
        delta=0.0,
        robust_norm=2,
        eps_i=1.0,
    )

    assert np.allclose(A_full[:2], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))
    assert np.allclose(b_full[:2], np.array([0.25, 0.5], dtype=np.float64))
    assert np.isclose(box_eps, 1.0)
    assert np.isclose(ball_eps, 1.0)


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
