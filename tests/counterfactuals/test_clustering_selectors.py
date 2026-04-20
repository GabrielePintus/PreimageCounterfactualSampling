import numpy as np
import pytest

from counterfactuals.utils.clustering import select_prototype_indices


def test_random_selector_is_reproducible_unique_and_capped():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)

    first = select_prototype_indices(X, 4, method="random", random_state=123)
    second = select_prototype_indices(X, 4, method="random", random_state=123)

    assert first.shape == (4,)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4
    assert np.all((0 <= first) & (first < len(X)))

    capped = select_prototype_indices(X, 99, method="random", random_state=123)
    assert np.array_equal(capped, np.arange(len(X)))


def test_boundary_random_selector_is_rejected_in_generic_clustering_utils():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)

    with pytest.raises(ValueError, match="Unknown prototype selection method 'boundary_random'"):
        select_prototype_indices(X, 4, method="boundary_random", random_state=123)
