import numpy as np
import pytest

from certcf.atlas import CVXPY_AVAILABLE, CertCFAtlas


def _make_atlas(norm, ohe_slices):
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.norm = norm
    atlas.ohe_slices = ohe_slices
    atlas.solver_maxiter = 200
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
    assert np.isclose(dist, np.linalg.norm(expected - x_query), atol=1e-7)
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
