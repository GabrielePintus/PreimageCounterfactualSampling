"""
Bounding Volume Hierarchy (BVH) for efficient spatial indexing of polytopes.

This module implements a BVH tree structure that enables O(log N) average-case
counterfactual search instead of O(N) linear scan, while preserving exact
coverage and certifiability guarantees.
"""

import heapq
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Dict


@dataclass
class BVHNode:
    """
    Node in a Bounding Volume Hierarchy.

    Each node stores an axis-aligned bounding box. Leaf nodes contain
    a single polytope index, while internal nodes have two children.

    Attributes
    ----------
    bbox_min : np.ndarray
        Lower corner of the bounding box, shape (d,).
    bbox_max : np.ndarray
        Upper corner of the bounding box, shape (d,).
    left : BVHNode, optional
        Left child node (None for leaves).
    right : BVHNode, optional
        Right child node (None for leaves).
    polytope_idx : int, optional
        Index of the polytope (only set for leaf nodes).
    """
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    left: Optional['BVHNode'] = None
    right: Optional['BVHNode'] = None
    polytope_idx: Optional[int] = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (contains a polytope)."""
        return self.polytope_idx is not None

    def distance_to_point(self, x: np.ndarray) -> float:
        """
        Compute minimum distance from point x to this bounding box.

        Parameters
        ----------
        x : np.ndarray
            Query point, shape (d,).

        Returns
        -------
        float
            Minimum Euclidean distance from x to the bounding box.
            Returns 0 if x is inside the box.
        """
        clamped = np.clip(x, self.bbox_min, self.bbox_max)
        return np.linalg.norm(x - clamped)


class BVHIndex:
    """
    Spatial index for certified polytopes using a Bounding Volume Hierarchy.

    This class builds and manages a BVH tree for efficient nearest-neighbor
    queries on a collection of polytopes.

    Attributes
    ----------
    root : BVHNode
        Root node of the BVH tree.
    n_polytopes : int
        Number of polytopes in the index.
    dimension : int
        Dimensionality of the input space.
    """

    def __init__(self, centers: np.ndarray, eps):
        """
        Build a BVH from polytope centers and perturbation radius.

        Parameters
        ----------
        centers : np.ndarray
            Array of polytope centers, shape (n_polytopes, d).
        eps : float or np.ndarray
            Perturbation radius defining each polytope's bounding box.  Can be
            a scalar (same radius for all) or a 1-D array of shape
            ``(n_polytopes,)`` for per-polytope radii.
        """
        self.centers = centers
        self.eps = eps
        self.n_polytopes = len(centers)
        self.dimension = centers.shape[1] if len(centers) > 0 else 0

        # Build the tree
        self.root = self._build_tree(centers, eps)

        # Compute tree statistics
        self._n_internal, self._n_leaves = self._count_nodes(self.root)

    def _build_tree(self, centers: np.ndarray, eps) -> BVHNode:
        """
        Recursively build the BVH tree.

        Uses median split along the axis with maximum spread for
        balanced tree construction.
        """
        n = len(centers)

        # Create leaf nodes
        leaves = []
        for i in range(n):
            center = centers[i]
            eps_i = eps[i] if isinstance(eps, np.ndarray) else eps
            leaves.append(BVHNode(
                bbox_min=center - eps_i,
                bbox_max=center + eps_i,
                polytope_idx=i
            ))

        return self._build_recursive(leaves)

    def _build_recursive(self, nodes: List[BVHNode]) -> BVHNode:
        """Recursively build tree from leaf nodes."""
        if len(nodes) == 1:
            return nodes[0]

        if len(nodes) == 2:
            left, right = nodes
            bbox_min = np.minimum(left.bbox_min, right.bbox_min)
            bbox_max = np.maximum(left.bbox_max, right.bbox_max)
            return BVHNode(
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                left=left,
                right=right
            )

        # Find axis with largest spread
        centers = np.array([(n.bbox_min + n.bbox_max) / 2 for n in nodes])
        spreads = centers.max(axis=0) - centers.min(axis=0)
        split_axis = np.argmax(spreads)

        # Sort by center along split axis and divide at median
        sorted_indices = np.argsort(centers[:, split_axis])
        mid = len(nodes) // 2

        left_nodes = [nodes[i] for i in sorted_indices[:mid]]
        right_nodes = [nodes[i] for i in sorted_indices[mid:]]

        left_child = self._build_recursive(left_nodes)
        right_child = self._build_recursive(right_nodes)

        bbox_min = np.minimum(left_child.bbox_min, right_child.bbox_min)
        bbox_max = np.maximum(left_child.bbox_max, right_child.bbox_max)

        return BVHNode(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            left=left_child,
            right=right_child
        )

    def _count_nodes(self, node: BVHNode) -> Tuple[int, int]:
        """Count internal and leaf nodes."""
        if node.is_leaf():
            return 0, 1
        left_i, left_l = self._count_nodes(node.left)
        right_i, right_l = self._count_nodes(node.right)
        return 1 + left_i + right_i, left_l + right_l

    @property
    def n_internal_nodes(self) -> int:
        """Number of internal nodes in the tree."""
        return self._n_internal

    @property
    def n_leaf_nodes(self) -> int:
        """Number of leaf nodes (equals n_polytopes)."""
        return self._n_leaves

    @property
    def tree_depth(self) -> int:
        """Maximum depth of the tree."""
        def depth(node):
            if node.is_leaf():
                return 1
            return 1 + max(depth(node.left), depth(node.right))
        return depth(self.root)

    def query_nearest(
        self,
        x: np.ndarray,
        project_fn: Callable[[int, float], Tuple[Optional[np.ndarray], float]],
        stats_out: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[np.ndarray], float, Optional[int], int]:
        """
        Find the nearest point in any polytope using branch-and-bound search.

        This method traverses the BVH, pruning subtrees that cannot contain
        a closer point than the current best, and only calls the projection
        function for polytopes that might improve the solution.

        Parameters
        ----------
        x : np.ndarray
            Query point, shape (d,).
        project_fn : callable
            Function that takes a polytope index and the current incumbent
            distance and returns (projected_point, distance).
            Should return (None, inf) if projection fails.

        Returns
        -------
        best_point : np.ndarray or None
            The closest point found, or None if no valid projection exists.
        best_dist : float
            Distance to the closest point (inf if none found).
        best_idx : int or None
            Index of the polytope containing the closest point.
        n_projections : int
            Number of projection operations performed (for benchmarking).
        """
        best_point = None
        best_dist = np.inf
        best_idx = None
        n_projections = 0
        n_nodes_popped = 0
        n_nodes_pruned = 0
        n_leaves_visited = 0
        max_queue_size = 1

        # Priority queue: (distance_to_bbox, unique_id, node)
        pq = [(self.root.distance_to_point(x), id(self.root), self.root)]

        while pq:
            max_queue_size = max(max_queue_size, len(pq))
            dist_to_bbox, _, node = heapq.heappop(pq)
            n_nodes_popped += 1

            # Pruning: skip if distance to bbox >= best found
            if dist_to_bbox >= best_dist:
                n_nodes_pruned += 1
                continue

            if node.is_leaf():
                n_leaves_visited += 1
                # Project onto this polytope
                point, dist = project_fn(node.polytope_idx, best_dist)
                n_projections += 1

                if dist < best_dist:
                    best_point = point
                    best_dist = dist
                    best_idx = node.polytope_idx
            else:
                # Add children to queue (with early pruning)
                for child in [node.left, node.right]:
                    if child is not None:
                        child_dist = child.distance_to_point(x)
                        if child_dist < best_dist:
                            heapq.heappush(pq, (child_dist, id(child), child))

        if stats_out is not None:
            stats_out["n_nodes_popped"] = float(n_nodes_popped)
            stats_out["n_nodes_pruned"] = float(n_nodes_pruned)
            stats_out["n_leaves_visited"] = float(n_leaves_visited)
            stats_out["max_queue_size"] = float(max_queue_size)
            stats_out["n_candidates_considered"] = float(n_projections)

        return best_point, best_dist, best_idx, n_projections

    def query_k_nearest_candidates(
        self,
        x: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Find k polytopes whose bounding boxes are closest to x.

        This is a fast heuristic that doesn't require projections,
        useful for quick candidate selection.

        Parameters
        ----------
        x : np.ndarray
            Query point, shape (d,).
        k : int
            Number of candidates to return.

        Returns
        -------
        list[int]
            Indices of the k nearest polytopes by bounding box distance.
        """
        # Simple implementation: compute distances to all centers
        # For small k, this is efficient enough
        dists = np.linalg.norm(self.centers - x, axis=1)
        return list(np.argsort(dists)[:k])

    def query_sorted_lower_bounds(
        self,
        x: np.ndarray,
        eps_array: np.ndarray,
        project_fn: Callable[[int, float], Tuple[Optional[np.ndarray], float]],
        atlas_norm: int = 2,
        stats_out: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[np.ndarray], float, Optional[int], int]:
        """Find the nearest polytope using a vectorised sorted lower-bound scan.

        For each polytope i, a lower bound on the projection distance is computed
        without solving any QP:

        - L2 / L1 atlas norm: lb_i = max(0, ||x - c_i||_2 - eps_i)
          [tight for L2 balls; valid for L1 since L1 ball ⊆ L2 ball]
        - L∞ atlas norm: lb_i = ||max(0, |x - c_i| - eps_i)||_2
          [exact for L∞ boxes — same formula as BVH but vectorised]

        Polytopes are sorted ascending by lb_i. The loop terminates as soon as
        lb_i >= best_dist, because all remaining polytopes are provably farther.

        For L2 atlas norm this gives tighter lower bounds than the BVH (which uses
        L∞ bounding boxes that are loose for L2 balls), so fewer QPs are solved.

        Parameters
        ----------
        x : np.ndarray
            Query point, shape (d,).
        eps_array : np.ndarray
            Per-polytope epsilon values, shape (n_polytopes,).
        project_fn : callable
            Function that takes a polytope index and the current incumbent
            distance and returns (projected_point, distance).
        atlas_norm : int or float
            The Lp norm used for the certification ball. Default 2.

        Returns
        -------
        best_point, best_dist, best_idx, n_projections
        """
        if atlas_norm == np.inf:
            # Exact tight lower bound for L∞ box: ||max(0, |x-c| - eps)||_2
            diff = np.abs(x[None, :] - self.centers) - eps_array[:, None]
            lower_bounds = np.linalg.norm(np.maximum(0.0, diff), axis=1)
        else:
            # L2 and L1: max(0, ||x-c||_2 - eps)
            center_dists = np.linalg.norm(self.centers - x, axis=1)
            lower_bounds = np.maximum(0.0, center_dists - eps_array)

        sorted_idx = np.argsort(lower_bounds)

        best_point: Optional[np.ndarray] = None
        best_dist = np.inf
        best_idx: Optional[int] = None
        n_projections = 0
        n_pruned_by_bound = 0
        best_lower_bound_at_termination = np.inf

        for i in sorted_idx:
            if lower_bounds[i] >= best_dist:
                n_pruned_by_bound = int(len(sorted_idx) - n_projections)
                best_lower_bound_at_termination = float(lower_bounds[i])
                break  # All remaining polytopes have lb >= best_dist — prune
            point, dist = project_fn(int(i), best_dist)
            n_projections += 1
            if dist < best_dist:
                best_dist = dist
                best_point = point
                best_idx = int(i)

        if n_pruned_by_bound == 0:
            n_pruned_by_bound = int(len(sorted_idx) - n_projections)
            if n_projections < len(sorted_idx):
                best_lower_bound_at_termination = float(lower_bounds[sorted_idx[n_projections]])

        if stats_out is not None:
            stats_out["n_candidates_considered"] = float(n_projections)
            stats_out["n_candidates_total"] = float(len(sorted_idx))
            stats_out["n_candidates_pruned_by_bound"] = float(n_pruned_by_bound)
            stats_out["best_lower_bound_at_termination"] = float(best_lower_bound_at_termination)

        return best_point, best_dist, best_idx, n_projections
