from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from certcf.atlas import CertCFAtlas
from certcf.geometry.operations import build_class_union
from certcf.geometry.polytopes import make_polygon
from certcf.visualization.plotting import plot_polytopes


def _simple_bounds(*, center_by_label: dict[int, np.ndarray], eps: float) -> dict[int, dict[str, np.ndarray]]:
    bounds: dict[int, dict[str, np.ndarray]] = {}
    for label, center in center_by_label.items():
        center = np.asarray(center, dtype=np.float64).reshape(1, 2)
        bounds[label] = {
            "lA": np.zeros((1, 1, 2), dtype=np.float64),
            "lbias": np.ones((1, 1), dtype=np.float64),
            "uA": np.zeros((1, 1, 2), dtype=np.float64),
            "ubias": np.ones((1, 1), dtype=np.float64),
            "X": center,
            "eps": np.array([eps], dtype=np.float64),
        }
    return bounds


def test_make_polygon_l1_uses_exact_diamond_not_outer_box():
    poly = make_polygon(
        np.zeros((1, 2), dtype=np.float64),
        np.ones(1, dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        eps=1.0,
        norm=1,
    )

    assert poly is not None
    assert poly.area == pytest.approx(2.0, abs=1e-9)
    assert poly.covers(Point(0.5, 0.25))
    assert not poly.covers(Point(0.75, 0.75))


def test_l1_class_unions_do_not_exhibit_false_cross_class_overlap():
    bounds = _simple_bounds(
        center_by_label={
            0: np.array([0.0, 0.0], dtype=np.float64),
            1: np.array([1.2, 1.2], dtype=np.float64),
        },
        eps=1.0,
    )

    union_0 = build_class_union(0, bounds, bounds[0]["eps"], norm=1)
    union_1 = build_class_union(1, bounds, bounds[1]["eps"], norm=1)

    overlap_point = np.array([0.6, 0.6], dtype=np.float64)
    assert np.max(np.abs(overlap_point - bounds[0]["X"][0])) <= 1.0
    assert np.max(np.abs(overlap_point - bounds[1]["X"][0])) <= 1.0

    assert union_0.intersection(union_1).area == pytest.approx(0.0, abs=1e-9)
    assert not union_0.covers(Point(*overlap_point))
    assert not union_1.covers(Point(*overlap_point))


def test_make_polygon_rejects_unsupported_2d_norms():
    with pytest.raises(ValueError, match="only supports L1 and L-inf"):
        make_polygon(
            np.zeros((1, 2), dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            eps=1.0,
            norm=2,
        )


def test_certcf_atlas_union_validation_rejects_cross_class_overlap():
    atlas = CertCFAtlas.__new__(CertCFAtlas)
    atlas._class_unions = {
        0: Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        1: Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),
    }

    with pytest.raises(ValueError, match="must be disjoint"):
        atlas._validate_class_unions(tol=1e-9)


def test_plot_polytopes_accepts_norm_argument():
    bounds = _simple_bounds(center_by_label={0: np.array([0.0, 0.0])}, eps=1.0)
    fig, ax = plot_polytopes(bounds, eps=1.0, n_classes=1, norm=1, figsize=(4, 4))
    assert fig is not None
    assert ax is not None
