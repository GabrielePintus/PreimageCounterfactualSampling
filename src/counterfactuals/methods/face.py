"""FACE method implementation for feasible and actionable counterfactuals.

This implementation follows the core algorithmic structure from
Poyiadzi et al. (2020):
- Build a weighted graph over observed instances.
- Filter candidate targets with prediction-confidence and density thresholds.
- Return the minimum-cost candidate reachable through shortest paths.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import radius_neighbors_graph

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface
from counterfactuals.density import BaseDensityEstimator, build_density_estimator
from counterfactuals.methods._constraints import (
    constraint_metadata,
    normalize_directional_dims,
    normalize_fixed_dims,
    satisfies_directional_constraints,
    validate_disjoint_directional_dims,
)


ConditionsFn = Callable[[np.ndarray, np.ndarray], bool]
WeightFn = Callable[[np.ndarray, np.ndarray], float]


class FACEMethod(BaseCounterfactualMethod):
    """Feasible and Actionable Counterfactual Explanations (FACE).

    Builds an ε-radius graph over training data; edge weights follow the paper's
    density-weighted distance formula. The density estimator is pluggable and
    determines both the density values used for weights and the candidate thresholds.
    """

    @staticmethod
    def _parse_density_estimator(config: Union[BaseDensityEstimator, dict]) -> BaseDensityEstimator:
        if isinstance(config, BaseDensityEstimator):
            return config
        elif isinstance(config, dict):
            return build_density_estimator(config["name"], **config.get("params", {}))
        else:
            raise ValueError("density_estimator must be either a BaseDensityEstimator instance or a config dict")

    @staticmethod
    def _normalize_norm(norm: int | float | str) -> int | float:
        if isinstance(norm, str):
            if norm.lower() in {"inf", "infinity"}:
                return np.inf
            return float(norm)
        return norm

    def __init__(
        self,
        # The trained model
        model: ModelInterface,

        # Main params of the FACE method
        density_estimator: Union[BaseDensityEstimator, dict],
        epsilon: float = float("inf"),
        norm: int | float | str = 2,
        tp: float = 0.5,
        td: float = 0.0,
        conditions_fn: Optional[ConditionsFn] = None,
        weight_fn: Optional[WeightFn] = None,
        fixed_dims: Optional[Sequence[int]] = None,
        immutable_features: Optional[Sequence[str]] = None,
        nondecreasing_dims: Optional[Sequence[int]] = None,
        nonincreasing_dims: Optional[Sequence[int]] = None,
        nondecreasing_features: Optional[Sequence[str]] = None,
        nonincreasing_features: Optional[Sequence[str]] = None,

        # Custom downsampling strategy, not mentioned in the original paper
        subsample_method: str = "kmedoids",
        k_per_class: Optional[int] = None,

        # Random seed for reproducibility (e.g., in subsampling)
        random_seed: int = 42,
    ):
        """Initialize the FACE method.
        Args:
            model: the classifier to explain
            density_estimator: a density estimator
            epsilon: threshold of closeness for two nodes to be neighbors
            norm: lp norm used for graph construction, edge distances, nearest-node projection,
                and the reported counterfactual distance. Supported values: 1, 2, "inf".
            tp: the model's prediction confidence threshold
            td: density threshold

            conditions_fn: per-query actionability filter; called as conditions_fn(x_query, x_candidate) to decide whether a candidate is a valid target for this specific query (e.g. "do not change feature sex")
            weight_fn: custom function to compute the edge weight between two nodes, by default it's based on density and distance as in the paper

            subsample_method: Method for downsampling the training data (e.g., "kmedoids", "kmeans", or None)
            k_per_class: Number of samples per class for downsampling
            random_seed: Random seed for reproducibility
        """
        super().__init__(model=model, random_seed=random_seed, k_per_class=k_per_class, subsample_method=subsample_method)

        self.density_estimator = FACEMethod._parse_density_estimator(density_estimator)
        self.epsilon = epsilon
        self.norm = self._normalize_norm(norm)
        self.tp = tp
        self.td = td
        self._base_conditions_fn = conditions_fn
        self.fixed_dims = normalize_fixed_dims(fixed_dims)
        self.immutable_features = tuple(str(name) for name in (immutable_features or ()))
        self.nondecreasing_dims = normalize_directional_dims(
            nondecreasing_dims,
            name="nondecreasing_dims",
        )
        self.nonincreasing_dims = normalize_directional_dims(
            nonincreasing_dims,
            name="nonincreasing_dims",
        )
        validate_disjoint_directional_dims(self.nondecreasing_dims, self.nonincreasing_dims)
        self.nondecreasing_features = tuple(str(name) for name in (nondecreasing_features or ()))
        self.nonincreasing_features = tuple(str(name) for name in (nonincreasing_features or ()))
        if weight_fn is not None:
            self.weight_fn = weight_fn

        self.train_proba: Optional[np.ndarray] = None
        self.density: Optional[np.ndarray] = None
        self.graph: Optional[nx.Graph] = None

    def _pairwise_metric_kwargs(self) -> dict:
        if self.norm == np.inf:
            return {"metric": "chebyshev"}
        return {"metric": "minkowski", "p": float(self.norm)}

    def _lp_distance(self, xi: np.ndarray, xj: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.linalg.norm(xi - xj, ord=self.norm, axis=axis)

    def conditions_fn(self, x_query: np.ndarray, x_candidate: np.ndarray) -> bool:
        """Per-query final-candidate actionability filter."""
        if self._base_conditions_fn is not None and not self._base_conditions_fn(x_query, x_candidate):
            return False
        if self.fixed_dims is not None and len(self.fixed_dims) > 0:
            if not np.allclose(x_candidate[self.fixed_dims], x_query[self.fixed_dims], atol=1e-6):
                return False
        if not satisfies_directional_constraints(
            x_candidate,
            x_query,
            self.nondecreasing_dims,
            self.nonincreasing_dims,
        ):
            return False
        return True

    def weight_fn(self, xi: np.ndarray, xj: np.ndarray) -> float:
        midpoint = ((xi + xj) / 2.0)
        distance = self._lp_distance(xi, xj)
        density = self.density_estimator(midpoint.reshape(1, -1))[0]

        # Avoid log(0) by adding a small epsilon to the density
        return -np.log(density) * distance if density > 1e-10 else float("inf")

    def _fit(self) -> None:
        # We start by fitting the density estimator and store all the density values for the training data
        self.density_estimator.fit(self._x_train)
        self.density = self.density_estimator(self._x_train).astype(np.float32)
        self.train_proba = np.asarray(self.model.predict_proba(self._x_train), dtype=np.float32)

        # Build the graph
        self.graph = self.build_graph(self._x_train)
        self._log_graph_stats(self.graph)

    def generate(self, x: np.ndarray, target_class: int) -> CounterfactualResult:
        if not self._is_fitted:
            raise RuntimeError("FACEMethod is not fitted. Call fit() before generate().")

        x_query = np.asarray(x, dtype=np.float32).reshape(1, -1)

        # Project query onto the graph via nearest neighbor respecting the starting label.
        query_label = int(self.model.predict(x_query)[0])
        predicted_train_labels = np.argmax(self.train_proba, axis=1)
        same_class_indices = np.where(predicted_train_labels == query_label)[0]
        local_idx = int(np.argmin(pairwise_distances(
            x_query,
            self._x_train[same_class_indices],
            **self._pairwise_metric_kwargs(),
        )[0]))
        start_node = int(same_class_indices[local_idx])

        # Compute the candidate nodes filtering by
        # - confidence threshold (tp) and
        # - density threshold (td)
        candidates = self.candidate_nodes(x_query=x_query[0], target_class=target_class)

        if len(candidates) == 0:
            # No candidates meet the confidence and density thresholds, return failure with reason.
            return CounterfactualResult(
                x_cf=x_query[0],
                success=False,
                distance=0.0,
                metadata={
                    "target_class": target_class,
                    "reason": "no_candidates",
                    "start_node": start_node,
                    **self._constraint_metadata(),
                },
            )

        # Compute shortest paths from the start node to all candidates, and select the best reachable one.
        try:
            path_cost, path = nx.multi_source_dijkstra(self.graph, sources=set(candidates), target=start_node, weight="weight")
        except nx.NetworkXNoPath:
            return CounterfactualResult(
                x_cf=x_query[0],
                success=False,
                distance=0.0,
                metadata={
                    "target_class": target_class,
                    "reason": "no_reachable_candidate",
                    "start_node": start_node,
                    "n_candidates": int(len(candidates)),
                    **self._constraint_metadata(),
                },
            )

        best_node = path[0]
        x_cf = self._x_train[best_node]
        success = int(self.model.predict(x_cf)[0]) == target_class

        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(success),
            distance=float(self._lp_distance(x_cf, x_query[0])),
            metadata={
                "target_class": target_class,
                "start_node": start_node,
                "target_node": int(best_node),
                "path_cost": float(path_cost),
                "path_indices": [int(i) for i in path],
                "n_candidates": int(len(candidates)),
                "norm": self.norm,
                **self._constraint_metadata(),
            },
        )

    def build_graph(self, x_train: np.ndarray) -> nx.Graph:
        # ε-radius graph: connect points within distance epsilon.
        # Set epsilon=inf for a complete graph (no pruning).
        adjacency = radius_neighbors_graph(
            x_train,
            radius=self.epsilon,
            mode="distance",
            include_self=False,
            **self._pairwise_metric_kwargs(),
        )
        graph = nx.from_scipy_sparse_array(adjacency)

        for i, j, _ in list(graph.edges(data=True)):
            xi = x_train[i]
            xj = x_train[j]

            graph[i][j]["weight"] = self.weight_fn(xi, xj)

        return graph


    def _log_graph_stats(self, graph: nx.Graph) -> None:
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        degrees = [d for _, d in graph.degree()]
        components = list(nx.connected_components(graph))
        n_components = len(components)
        n_isolated = sum(1 for c in components if len(c) == 1)
        largest_cc = max(len(c) for c in components) if components else 0

        avg_deg = float(np.mean(degrees)) if degrees else 0.0
        med_deg = float(np.median(degrees)) if degrees else 0.0
        max_deg = int(max(degrees)) if degrees else 0

        print(
            f"[FACE graph] nodes={n_nodes}  edges={n_edges}  "
            f"degree avg={avg_deg:.1f} median={med_deg:.0f} max={max_deg}  "
            f"connected_components={n_components}  isolated_nodes={n_isolated}  "
            f"largest_component={largest_cc} ({100*largest_cc/n_nodes:.1f}%)"
        )

    def candidate_nodes(self, x_query: np.ndarray, target_class: int) -> np.ndarray:
        conf_mask    = self.train_proba[:, target_class] >= self.tp
        density_mask = self.density >= self.td
        indices = np.where(conf_mask & density_mask)[0]
        return np.array([i for i in indices if self.conditions_fn(x_query, self._x_train[i])])

    def _constraint_metadata(self) -> dict[str, int | str]:
        return constraint_metadata(
            self.fixed_dims,
            self.immutable_features,
            self.nondecreasing_dims,
            self.nonincreasing_dims,
            self.nondecreasing_features,
            self.nonincreasing_features,
        )
