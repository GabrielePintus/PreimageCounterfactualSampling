"""FACE method implementation for feasible and actionable counterfactuals.

This implementation follows the core algorithmic structure from
Poyiadzi et al. (2020):
- Build a weighted graph over observed instances.
- Filter candidate targets with prediction-confidence and density thresholds.
- Return the minimum-cost candidate reachable through shortest paths.
"""

from __future__ import annotations

from typing import Callable, Optional

import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity, NearestNeighbors, kneighbors_graph, radius_neighbors_graph

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface


ActionabilityFn = Callable[[np.ndarray, np.ndarray], bool]
CostFn = Callable[[np.ndarray, np.ndarray], float]
EmbeddingFn = Callable[[np.ndarray], np.ndarray]


class FACEMethod(BaseCounterfactualMethod):
    """Feasible and Actionable Counterfactual Explanations (FACE).

    Parameters mirror the paper's core controls:
    - graph construction (`knn` or `epsilon`),
    - confidence threshold (`tp`),
    - density threshold (`td`).
    """

    def __init__(
        self,
        graph_mode: str = "knn",
        n_neighbors: int = 15,
        epsilon: float = 0.5,
        tp: float = 0.5,
        td: float = 0.0,
        density_bandwidth: float = 0.5,
        actionability_fn: Optional[ActionabilityFn] = None,
        cost_fn: Optional[CostFn] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        if graph_mode not in {"knn", "epsilon"}:
            raise ValueError("graph_mode must be one of {'knn', 'epsilon'}")
        self.graph_mode = graph_mode
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon
        self.tp = tp
        self.td = td
        self.density_bandwidth = density_bandwidth
        self.actionability_fn = actionability_fn
        self.cost_fn = cost_fn
        self.embedding_fn = embedding_fn

        self._x_train: Optional[np.ndarray] = None
        self._z_train: Optional[np.ndarray] = None
        self._train_proba: Optional[np.ndarray] = None
        self._density: Optional[np.ndarray] = None
        self._kde: Optional[KernelDensity] = None
        self._graph: Optional[nx.Graph] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        del y_train
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._z_train = self._embed(self._x_train)

        self._kde = KernelDensity(kernel="gaussian", bandwidth=self.density_bandwidth)
        self._kde.fit(self._z_train)
        self._density = np.exp(self._kde.score_samples(self._z_train)).astype(np.float32)

        self._train_proba = np.asarray(model.predict_proba(self._x_train), dtype=np.float32)
        self._graph = self._build_graph(self._z_train)
        self._is_fitted = True

    def generate(self, example: CounterfactualExample, model: ModelInterface) -> CounterfactualResult:
        if not self._is_fitted:
            raise RuntimeError("FACEMethod is not fitted. Call fit() before generate().")
        assert self._x_train is not None
        assert self._z_train is not None
        assert self._train_proba is not None
        assert self._density is not None
        assert self._graph is not None

        x_query = np.asarray(example.x, dtype=np.float32).reshape(1, -1)
        z_query = self._embed(x_query)
        target_class = self._resolve_target_class(example=example, model=model)
        start_node = int(np.argmin(pairwise_distances(z_query, self._z_train)[0]))

        candidates = self._candidate_nodes(target_class=target_class)
        if len(candidates) == 0:
            return CounterfactualResult(
                x_cf=x_query[0],
                success=False,
                distance=0.0,
                metadata={"target_class": target_class, "reason": "no_candidates", "start_node": start_node},
            )

        lengths, paths = nx.single_source_dijkstra(self._graph, start_node, weight="weight")
        reachable = [node for node in candidates if node in lengths]
        if len(reachable) == 0:
            return CounterfactualResult(
                x_cf=x_query[0],
                success=False,
                distance=0.0,
                metadata={"target_class": target_class, "reason": "no_reachable_candidate", "start_node": start_node},
            )

        best_node = min(reachable, key=lambda n: float(lengths[n]))
        path_indices = [int(i) for i in paths[best_node]]
        x_cf = self._x_train[best_node]
        success = int(model.predict(x_cf)[0]) == target_class

        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(success),
            distance=float(np.linalg.norm(x_cf - x_query[0], ord=2)),
            metadata={
                "target_class": target_class,
                "start_node": start_node,
                "target_node": int(best_node),
                "path_cost": float(lengths[best_node]),
                "path_indices": path_indices,
                "n_candidates": int(len(candidates)),
            },
        )

    def _build_graph(self, z_train: np.ndarray) -> nx.Graph:
        if self.graph_mode == "knn":
            adjacency = kneighbors_graph(
                z_train,
                n_neighbors=self.n_neighbors,
                mode="distance",
                include_self=False,
            )
        else:
            nn = NearestNeighbors(radius=self.epsilon, metric="euclidean")
            nn.fit(z_train)
            adjacency = radius_neighbors_graph(nn, z_train, radius=self.epsilon, mode="distance", include_self=False)

        graph = nx.from_scipy_sparse_array(adjacency)

        # Re-weight edges with inverse midpoint density, as in density-weighted shortest paths.
        for i, j, attrs in list(graph.edges(data=True)):
            xi = self._x_train[i]
            xj = self._x_train[j]
            zi = z_train[i]
            zj = z_train[j]
            dist = float(attrs.get("weight", np.linalg.norm(zi - zj, ord=2)))

            if self.actionability_fn is not None and not self.actionability_fn(xi, xj):
                graph.remove_edge(i, j)
                continue

            # Fast midpoint density proxy: geometric mean of endpoint densities.
            # This keeps density-weighted shortest paths practical on larger graphs.
            p_i = float(self._density[i]) if self._density is not None else 1.0
            p_j = float(self._density[j]) if self._density is not None else 1.0
            p_mid = float(np.sqrt(max(p_i, 1e-12) * max(p_j, 1e-12)))
            edge_cost = float(dist / max(p_mid, 1e-12))
            if self.cost_fn is not None:
                edge_cost = float(self.cost_fn(xi, xj))

            graph[i][j]["weight"] = edge_cost
        return graph

    def _candidate_nodes(self, target_class: int) -> np.ndarray:
        assert self._train_proba is not None
        assert self._density is not None
        conf_mask = self._train_proba[:, target_class] >= float(self.tp)
        density_mask = self._density >= float(self.td)
        return np.where(conf_mask & density_mask)[0]

    def _embed(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if self.embedding_fn is None:
            return arr
        embedded = np.asarray(self.embedding_fn(arr), dtype=np.float32)
        return embedded

    @staticmethod
    def _resolve_target_class(example: CounterfactualExample, model: ModelInterface) -> int:
        if example.target_class is not None:
            return int(example.target_class)
        pred = int(model.predict(example.x)[0])
        proba = model.predict_proba(example.x)
        if proba.shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
