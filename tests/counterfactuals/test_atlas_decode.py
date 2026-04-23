import numpy as np
import pytest

from certcf.atlas import CVXPY_AVAILABLE, CertCFAtlas


def _make_atlas(norm, ohe_slices):
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.norm = norm
    atlas.distance_norm = norm
    atlas.ohe_slices = ohe_slices
    atlas.solver_maxiter = 200
    atlas.cvxpy_solvers = ["CLARABEL", "SCS"]
    atlas.cvxpy_solver_options = {}
    atlas.cvxpy_accept_statuses = {
        "CLARABEL": ["optimal"],
        "SCS": ["optimal", "optimal_inaccurate"],
    }
    atlas.ohe_decode_mode = "exact"
    atlas.decode_beam_width = 8
    atlas.decode_beam_branch_top_k = 3
    atlas.decode_beam_max_solver_calls = 32
    return atlas


def _two_block_decode_problem(extra_tail=None):
    tail = [] if extra_tail is None else [extra_tail]
    x_query = np.array([0.6, 0.4, 0.6, 0.4, *tail], dtype=np.float64)
    center = x_query.copy()
    A_rows = [
        [0.0, 1.0, 0.0, 0.0, *([0.0] if extra_tail is not None else [])],
        [0.0, 0.0, 0.0, 1.0, *([0.0] if extra_tail is not None else [])],
    ]
    A_full = np.array(A_rows, dtype=np.float64)
    b_full = np.array([-0.2, -0.2], dtype=np.float64)
    target = np.array([0.0, 1.0, 0.0, 1.0, *tail], dtype=np.float64)
    return x_query, center, A_full, b_full, target


def _decode_ball_eps(norm) -> float:
    return 3.0 if norm == 1 else 2.0


@pytest.mark.parametrize("norm", [1, 2])
def test_exact_enum_recovers_certified_vertex_missed_by_heuristic(norm):
    if not CVXPY_AVAILABLE:
        pytest.skip("CVXPY is required to exercise the L1/L2 exact decode path.")

    atlas = _make_atlas(norm, [(0, 2), (2, 4)])
    x_query, center, A_full, b_full, expected = _two_block_decode_problem()
    ball_eps = _decode_ball_eps(norm)

    heuristic_x, heuristic_dist = atlas._heuristic_polytope_decode(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        ball_eps=ball_eps,
        tol=1e-9,
    )

    assert heuristic_x is None
    assert np.isinf(heuristic_dist)

    x_cf, dist, profile = atlas._polytope_aware_decode(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=ball_eps,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
    )

    assert x_cf is not None
    assert np.allclose(x_cf, expected, atol=1e-7)
    assert np.isclose(dist, np.linalg.norm(expected - x_query, ord=atlas.distance_norm), atol=1e-7)
    assert profile["decode_mode"] == "exact_enum"
    assert profile["decode_exact_fallback_used"] is True
    assert profile["decode_heuristic_success"] is False


def test_exact_enum_returns_none_when_no_certified_discrete_vertex_exists():
    atlas = _make_atlas(np.inf, [(0, 2)])
    x_query = np.array([0.5, 0.5], dtype=np.float64)
    center = x_query.copy()
    A_full = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b_full = np.array([-0.3, -0.3], dtype=np.float64)

    x_cf, dist, profile = atlas._polytope_aware_decode(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
    )

    assert x_cf is None
    assert np.isinf(dist)
    assert profile["decode_mode"] == "exact_enum"
    assert profile["decode_exact_fallback_used"] is True
    assert profile["decode_solver_calls"] == 2


def test_branch_and_bound_matches_exact_enumeration():
    if not CVXPY_AVAILABLE:
        pytest.skip("CVXPY is required to exercise the L2 exact decode path.")

    atlas = _make_atlas(2, [(0, 2), (2, 4)])
    x_query, center, A_full, b_full, _ = _two_block_decode_problem()
    ball_eps = _decode_ball_eps(2)

    enum_x, enum_dist, enum_profile = atlas._exact_polytope_decode_enumeration(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=ball_eps,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        incumbent_upper_bound=np.inf,
    )
    bnb_x, bnb_dist, bnb_profile = atlas._exact_polytope_decode_branch_and_bound(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=ball_eps,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        incumbent_upper_bound=np.inf,
    )

    assert enum_x is not None and bnb_x is not None
    assert np.allclose(bnb_x, enum_x, atol=1e-7)
    assert np.isclose(bnb_dist, enum_dist, atol=1e-7)
    assert bnb_profile["decode_mode"] == "exact_bnb"
    assert bnb_profile["decode_solver_calls"] <= enum_profile["decode_solver_calls"]


def test_exact_decode_respects_fixed_dimensions():
    if not CVXPY_AVAILABLE:
        pytest.skip("CVXPY is required to exercise the L2 exact decode path.")

    atlas = _make_atlas(2, [(0, 2), (2, 4)])
    x_query, center, A_full, b_full, expected = _two_block_decode_problem(extra_tail=0.75)
    fixed_dims = np.array([4], dtype=np.int64)
    ball_eps = _decode_ball_eps(2)

    x_cf, dist, profile = atlas._polytope_aware_decode(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=ball_eps,
        maxiter=200,
        tol=1e-9,
        fixed_dims=fixed_dims,
    )

    assert x_cf is not None
    assert np.allclose(x_cf, expected, atol=1e-7)
    assert np.isclose(x_cf[fixed_dims[0]], x_query[fixed_dims[0]], atol=1e-9)
    assert np.isclose(dist, np.linalg.norm(expected - x_query), atol=1e-7)
    assert profile["decode_exact_fallback_used"] is True


def test_exact_decode_uses_incumbent_bound_for_early_pruning():
    atlas = _make_atlas(np.inf, [(0, 2)])
    x_query = np.array([0.8, 0.2], dtype=np.float64)
    x_star = np.array([0.5, 0.5], dtype=np.float64)
    center = x_star.copy()
    A_full = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    b_full = np.array([-0.1, -0.1], dtype=np.float64)

    x_cf, dist, profile = atlas._exact_polytope_decode_enumeration(
        x_star,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        incumbent_upper_bound=0.1,
    )

    assert x_cf is None
    assert np.isinf(dist)
    assert profile["decode_nodes_pruned"] == 1
    assert profile["decode_solver_calls"] == 0


@pytest.mark.parametrize("norm", [1, 2])
def test_beam_decode_recovers_certified_vertex_missed_by_heuristic(norm):
    if not CVXPY_AVAILABLE:
        pytest.skip("CVXPY is required to exercise the L1/L2 beam decode path.")

    atlas = _make_atlas(norm, [(0, 2), (2, 4)])
    atlas.ohe_decode_mode = "beam_only"
    atlas.decode_beam_width = 4
    atlas.decode_beam_branch_top_k = 2
    atlas.decode_beam_max_solver_calls = 8
    x_query, center, A_full, b_full, expected = _two_block_decode_problem()
    ball_eps = _decode_ball_eps(norm)

    x_cf, dist, profile = atlas._polytope_aware_decode(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=ball_eps,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
    )

    assert x_cf is not None
    assert np.allclose(x_cf, expected, atol=1e-7)
    assert np.isclose(dist, np.linalg.norm(expected - x_query, ord=atlas.distance_norm), atol=1e-7)
    assert profile["decode_mode"] == "beam"
    assert profile["decode_exact_fallback_used"] is False
    assert profile["decode_beam_attempted"] is True
    assert profile["decode_beam_fallback_to_exact"] is False


def test_beam_then_exact_falls_back_to_exact_decode_when_beam_fails(monkeypatch):
    atlas = _make_atlas(np.inf, [(0, 2)])
    atlas.ohe_decode_mode = "beam_then_exact"
    beam_profile = atlas._build_decode_profile(
        mode="beam",
        exact_fallback_used=False,
        nodes_visited=1,
        nodes_pruned=1,
        solver_calls=1,
        product_size=2,
        heuristic_success=False,
        beam_attempted=True,
        beam_budget_exhausted=True,
    )
    exact_profile = atlas._build_decode_profile(
        mode="exact_enum",
        exact_fallback_used=True,
        nodes_visited=2,
        nodes_pruned=0,
        solver_calls=2,
        product_size=2,
        heuristic_success=False,
    )

    monkeypatch.setattr(atlas, "_heuristic_polytope_decode", lambda *args, **kwargs: (None, np.inf))
    monkeypatch.setattr(atlas, "_beam_polytope_decode", lambda *args, **kwargs: (None, np.inf, dict(beam_profile)))
    monkeypatch.setattr(
        atlas,
        "_exact_polytope_decode_enumeration",
        lambda *args, **kwargs: (np.array([1.0, 0.0]), 1.0, dict(exact_profile)),
    )

    x_cf, dist, profile = atlas._polytope_aware_decode(
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
    )

    assert x_cf is not None
    assert np.isclose(dist, 1.0)
    assert profile["decode_mode"] == "exact_enum"
    assert profile["decode_exact_fallback_used"] is True
    assert profile["decode_beam_attempted"] is True
    assert profile["decode_beam_budget_exhausted"] is True
    assert profile["decode_beam_fallback_to_exact"] is True


def test_beam_only_returns_failure_without_exact_fallback_when_beam_fails(monkeypatch):
    atlas = _make_atlas(np.inf, [(0, 2)])
    atlas.ohe_decode_mode = "beam_only"
    beam_profile = atlas._build_decode_profile(
        mode="beam",
        exact_fallback_used=False,
        nodes_visited=1,
        nodes_pruned=1,
        solver_calls=1,
        product_size=2,
        heuristic_success=False,
        beam_attempted=True,
    )

    monkeypatch.setattr(atlas, "_heuristic_polytope_decode", lambda *args, **kwargs: (None, np.inf))
    monkeypatch.setattr(atlas, "_beam_polytope_decode", lambda *args, **kwargs: (None, np.inf, dict(beam_profile)))
    monkeypatch.setattr(
        atlas,
        "_exact_polytope_decode_enumeration",
        lambda *args, **kwargs: pytest.fail("exact enumeration should not run in beam_only mode"),
    )
    monkeypatch.setattr(
        atlas,
        "_exact_polytope_decode_branch_and_bound",
        lambda *args, **kwargs: pytest.fail("branch-and-bound should not run in beam_only mode"),
    )

    x_cf, dist, profile = atlas._polytope_aware_decode(
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.eye(2, dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
    )

    assert x_cf is None
    assert np.isinf(dist)
    assert profile["decode_mode"] == "beam"
    assert profile["decode_exact_fallback_used"] is False
    assert profile["decode_beam_attempted"] is True
    assert profile["decode_beam_fallback_to_exact"] is False


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required to exercise the L2 beam decode path.")
def test_beam_decode_respects_fixed_dimensions():
    atlas = _make_atlas(2, [(0, 2), (2, 4)])
    atlas.ohe_decode_mode = "beam_only"
    x_query, center, A_full, b_full, expected = _two_block_decode_problem(extra_tail=0.75)
    fixed_dims = np.array([4], dtype=np.int64)
    ball_eps = _decode_ball_eps(2)

    x_cf, dist, profile = atlas._polytope_aware_decode(
        x_query,
        x_query,
        A_full,
        b_full,
        center,
        box_eps=1.0,
        ball_eps=ball_eps,
        maxiter=200,
        tol=1e-9,
        fixed_dims=fixed_dims,
    )

    assert x_cf is not None
    assert np.allclose(x_cf, expected, atol=1e-7)
    assert np.isclose(x_cf[fixed_dims[0]], x_query[fixed_dims[0]], atol=1e-9)
    assert np.isclose(dist, np.linalg.norm(expected - x_query), atol=1e-7)
    assert profile["decode_mode"] == "beam"


def test_beam_decode_respects_branch_top_k():
    atlas = _make_atlas(np.inf, [(0, 3)])
    atlas.ohe_decode_mode = "beam_only"
    atlas.decode_beam_branch_top_k = 2
    visited_categories = []

    def fake_solve(
        x0,
        A_full,
        b_full,
        center,
        box_eps,
        ball_eps,
        maxiter,
        tol,
        fixed_dims=None,
        ohe_slices=None,
        fixed_ohe_assignments=None,
        profile_out=None,
    ):
        del x0, A_full, b_full, center, box_eps, ball_eps, maxiter, tol, fixed_dims, ohe_slices, profile_out
        category = fixed_ohe_assignments[0]
        visited_categories.append(category)
        child = np.zeros(3, dtype=np.float64)
        child[category] = 1.0
        return child, float(category)

    atlas._solve_projection_subproblem = fake_solve
    x_cf, dist, profile = atlas._beam_polytope_decode(
        x_star=np.array([0.6, 0.3, 0.1], dtype=np.float64),
        x_query=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        A_full=np.eye(3, dtype=np.float64),
        b_full=np.zeros(3, dtype=np.float64),
        center=np.zeros(3, dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        incumbent_upper_bound=np.inf,
    )

    assert x_cf is not None
    assert np.isclose(dist, 0.0)
    assert visited_categories == [0, 1]
    assert profile["decode_solver_calls"] == 2


def test_beam_decode_respects_beam_width():
    atlas = _make_atlas(np.inf, [(0, 3), (3, 5)])
    atlas.ohe_decode_mode = "beam_only"
    atlas.decode_beam_width = 2
    atlas.decode_beam_branch_top_k = 3
    expanded_second_layer = []

    def fake_solve(
        x0,
        A_full,
        b_full,
        center,
        box_eps,
        ball_eps,
        maxiter,
        tol,
        fixed_dims=None,
        ohe_slices=None,
        fixed_ohe_assignments=None,
        profile_out=None,
    ):
        del x0, A_full, b_full, center, box_eps, ball_eps, maxiter, tol, fixed_dims, ohe_slices, profile_out
        assignments = dict(fixed_ohe_assignments)
        if len(assignments) == 1:
            first_choice = assignments[0]
            child = np.array([0.0, 0.0, 0.0, 0.9, 0.1], dtype=np.float64)
            child[first_choice] = 1.0
            return child, float({0: 3.0, 1: 1.0, 2: 2.0}[first_choice])
        expanded_second_layer.append(assignments[0])
        child = np.zeros(5, dtype=np.float64)
        child[assignments[0]] = 1.0
        child[3 + assignments[1]] = 1.0
        return child, float(10 + assignments[0] + assignments[1])

    atlas._solve_projection_subproblem = fake_solve
    x_cf, _, profile = atlas._beam_polytope_decode(
        x_star=np.array([0.6, 0.3, 0.1, 0.9, 0.1], dtype=np.float64),
        x_query=np.zeros(5, dtype=np.float64),
        A_full=np.eye(5, dtype=np.float64),
        b_full=np.zeros(5, dtype=np.float64),
        center=np.zeros(5, dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        incumbent_upper_bound=np.inf,
    )

    assert x_cf is not None
    assert expanded_second_layer == [1, 1, 2, 2]
    assert profile["decode_solver_calls"] == 7


def test_beam_decode_respects_solver_call_budget():
    atlas = _make_atlas(np.inf, [(0, 3), (3, 5)])
    atlas.ohe_decode_mode = "beam_only"
    atlas.decode_beam_branch_top_k = 3
    atlas.decode_beam_max_solver_calls = 2
    calls = []

    def fake_solve(
        x0,
        A_full,
        b_full,
        center,
        box_eps,
        ball_eps,
        maxiter,
        tol,
        fixed_dims=None,
        ohe_slices=None,
        fixed_ohe_assignments=None,
        profile_out=None,
    ):
        del x0, A_full, b_full, center, box_eps, ball_eps, maxiter, tol, fixed_dims, ohe_slices, profile_out
        calls.append(dict(fixed_ohe_assignments))
        return np.ones(5, dtype=np.float64), 1.0

    atlas._solve_projection_subproblem = fake_solve
    x_cf, dist, profile = atlas._beam_polytope_decode(
        x_star=np.array([0.6, 0.3, 0.1, 0.55, 0.45], dtype=np.float64),
        x_query=np.zeros(5, dtype=np.float64),
        A_full=np.eye(5, dtype=np.float64),
        b_full=np.zeros(5, dtype=np.float64),
        center=np.zeros(5, dtype=np.float64),
        box_eps=1.0,
        ball_eps=1.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        incumbent_upper_bound=np.inf,
    )

    assert x_cf is None
    assert np.isinf(dist)
    assert len(calls) == 2
    assert profile["decode_solver_calls"] == 2
    assert profile["decode_beam_budget_exhausted"] is True
