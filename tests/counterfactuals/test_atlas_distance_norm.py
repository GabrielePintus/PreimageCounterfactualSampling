import numpy as np
import pytest

from certcf.atlas import CVXPY_AVAILABLE, CertCFAtlas
from certcf.indexing.bvh import BVHNode


def _make_atlas(cert_norm, distance_norm):
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas.norm = cert_norm
    atlas.distance_norm = distance_norm
    atlas.ohe_slices = None
    atlas.solver_maxiter = 200
    return atlas


def test_bvh_node_distance_to_point_uses_requested_norm():
    node = BVHNode(
        bbox_min=np.array([0.0, 0.0], dtype=np.float64),
        bbox_max=np.array([1.0, 1.0], dtype=np.float64),
    )
    x = np.array([2.0, 3.0], dtype=np.float64)

    assert np.isclose(node.distance_to_point(x, distance_norm=1), 3.0)
    assert np.isclose(node.distance_to_point(x, distance_norm=2), np.sqrt(5.0))
    assert np.isclose(node.distance_to_point(x, distance_norm=np.inf), 2.0)


@pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY is required for norm-aware distance objectives.")
@pytest.mark.parametrize(
    ("distance_norm", "expected"),
    [
        (1, np.array([0.0, 0.5], dtype=np.float64)),
        (2, np.array([0.2, 0.4], dtype=np.float64)),
        (np.inf, np.array([1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)),
    ],
)
def test_projection_uses_selected_distance_norm(distance_norm, expected):
    atlas = _make_atlas(np.inf, distance_norm)
    x_query = np.array([0.0, 0.0], dtype=np.float64)
    center = np.array([0.0, 0.0], dtype=np.float64)
    A_full = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    b_full = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    x_proj, dist = atlas._solve_projection_subproblem(
        x_query,
        A_full,
        b_full,
        center,
        box_eps=2.0,
        ball_eps=2.0,
        maxiter=200,
        tol=1e-9,
        fixed_dims=None,
        ohe_slices=None,
        fixed_ohe_assignments=None,
    )

    assert x_proj is not None
    assert np.allclose(x_proj, expected, atol=1e-5)
    assert np.isclose(dist, np.linalg.norm(x_proj - x_query, ord=distance_norm), atol=1e-6)
