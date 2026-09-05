import numpy as np

from verix import (
    deletion_transform,
    occlusion_sensitivity_order,
    random_order,
    reversal_transform,
)


def test_sensitivity_traversal_orders_from_least_to_most_sensitive():
    weights = np.array([1.0, 3.0, -2.0])

    def scores(batch):
        class_one = np.asarray(batch) @ weights
        return np.stack([-class_one, class_one], axis=1)

    x = np.array([1.0, 1.0, 1.0])
    order, sensitivity = occlusion_sensitivity_order(
        x,
        score_function=scores,
        predicted_class=1,
        transform=deletion_transform,
    )

    assert np.allclose(sensitivity, [1.0, 3.0, -2.0])
    assert order == (2, 0, 1)


def test_sensitivity_traversal_supports_grouped_features():
    def scores(batch):
        class_one = np.asarray(batch).sum(axis=1)
        return np.stack([-class_one, class_one], axis=1)

    order, sensitivity = occlusion_sensitivity_order(
        np.array([0.2, 0.3, 0.4]),
        score_function=scores,
        predicted_class=1,
        transform=reversal_transform(upper_bounds=1.0),
        feature_groups=[(0,), (1, 2)],
    )

    assert np.allclose(sensitivity, [-0.6, -0.6])
    assert order == (0, 1)


def test_random_order_matches_seeded_reference_behavior():
    assert random_order(6, seed=7) == random_order(6, seed=7)
    assert sorted(random_order(6, seed=7)) == list(range(6))
