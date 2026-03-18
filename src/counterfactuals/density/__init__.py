from .base import BaseDensityEstimator
from .estimators import EpsilonBallEstimator, KDEEstimator, KNNEstimator

ESTIMATORS: dict = {
    "kde": KDEEstimator,
    "knn": KNNEstimator,
    "epsilon": EpsilonBallEstimator,
}


def build_density_estimator(name: str, **params) -> BaseDensityEstimator:
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown density estimator '{name}'. Available: {list(ESTIMATORS)}")
    return ESTIMATORS[name](**params)


__all__ = [
    "BaseDensityEstimator",
    "KDEEstimator",
    "KNNEstimator",
    "EpsilonBallEstimator",
    "ESTIMATORS",
    "build_density_estimator",
]
