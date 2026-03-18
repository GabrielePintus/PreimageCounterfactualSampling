"""FACE method implementation for feasible and actionable counterfactuals.

This implementation follows the core algorithmic structure from
Poyiadzi et al. (2020):
- Build a weighted graph over observed instances.
- Filter candidate targets with prediction-confidence and density thresholds.
- Return the minimum-cost candidate reachable through shortest paths.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import radius_neighbors_graph

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface
from counterfactuals.density import BaseDensityEstimator, build_density_estimator


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

    def __init__(
        self,
        model: ModelInterface,

        # Main params of the FACE method
        density_estimator: Union[BaseDensityEstimator, dict],
        epsilon: float = float("inf"),
        tp: float = 0.5,
        td: float = 0.0,
        conditions_fn: Optional[ConditionsFn] = None,
        weight_fn: Optional[WeightFn] = None,

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
            epsilon: threshold of closeness for two nodes to be neighbors. the paper uses euclidean norm
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
        self.tp = tp
        self.td = td
        if conditions_fn is not None:
            self.conditions_fn = conditions_fn
        if weight_fn is not None:
            self.weight_fn = weight_fn

        self.train_proba: Optional[np.ndarray] = None
        self.density: Optional[np.ndarray] = None
        self.graph: Optional[nx.Graph] = None

    def conditions_fn(self, x_query: np.ndarray, x_candidate: np.ndarray) -> bool:
        """Default conditions function that allows all candidates."""
        return True

    def weight_fn(self, xi: np.ndarray, xj: np.ndarray) -> float:
        midpoint = ((xi + xj) / 2.0)
        distance = np.linalg.norm(xi - xj, ord=2)
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
        local_idx = int(np.argmin(pairwise_distances(x_query, self._x_train[same_class_indices])[0]))
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
                metadata={"target_class": target_class, "reason": "no_candidates", "start_node": start_node},
            )

        # Compute shortest paths from the start node to all candidates, and select the best reachable one.
        try:
            path_cost, path = nx.multi_source_dijkstra(self.graph, sources=set(candidates), target=start_node, weight="weight")
        except nx.NetworkXNoPath:
            return CounterfactualResult(
                x_cf=x_query[0],
                success=False,
                distance=0.0,
                metadata={"target_class": target_class, "reason": "no_reachable_candidate", "start_node": start_node},
            )

        best_node = path[0]
        x_cf = self._x_train[best_node]
        success = int(self.model.predict(x_cf)[0]) == target_class

        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(success),
            distance=float(np.linalg.norm(x_cf - x_query[0], ord=2)),
            metadata={
                "target_class": target_class,
                "start_node": start_node,
                "target_node": int(best_node),
                "path_cost": float(path_cost),
                "path_indices": [int(i) for i in path],
                "n_candidates": int(len(candidates)),
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



