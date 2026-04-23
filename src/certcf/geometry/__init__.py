"""Geometric operations for certified polytopes."""

from .polytopes import make_polygon, ball_box_constraints, exact_2d_ball_constraints
from .conversion import halfspace_to_vertices
from .operations import build_class_union, compute_overlap_matrix, assert_no_cross_class_overlap

__all__ = [
    "make_polygon",
    "ball_box_constraints",
    "exact_2d_ball_constraints",
    "halfspace_to_vertices",
    "build_class_union",
    "compute_overlap_matrix",
    "assert_no_cross_class_overlap",
]
