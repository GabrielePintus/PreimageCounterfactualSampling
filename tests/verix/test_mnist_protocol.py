import numpy as np

from verix.core import CounterfactualWitness
from verix.mnist import (
    WitnessValidation,
    select_nearest_valid_witness,
)


def validation(index, feature, l1, target, valid=True):
    return WitnessValidation(
        witness_index=index,
        feature=feature,
        predicted_class=target,
        class_changed=True,
        in_bounds=True,
        free_linf_distance=0.05,
        fixed_max_abs_difference=0.0,
        l1_distance=l1,
        l2_distance=l1,
        l0_changed=1,
        target_logit_margin=0.1,
        canonical_valid=valid,
    )


def witness(feature, value):
    return CounterfactualWitness(
        feature=feature,
        free_features=(feature,),
        free_input_indices=(feature,),
        x=np.full(784, value, dtype=np.float64),
        output=1,
    )


def test_candidate_selection_uses_nearest_valid_l1_witness():
    witnesses = [witness(4, 0.1), witness(8, 0.2), witness(3, 0.3)]
    validations = [
        validation(0, 4, 3.0, 2),
        validation(1, 8, 1.0, 7),
        validation(2, 3, 0.5, 9, valid=False),
    ]

    selected = select_nearest_valid_witness(witnesses, validations)

    assert selected is not None
    assert selected.witness_index == 1
    assert selected.feature == 8
    assert selected.target_class == 7


def test_candidate_selection_has_deterministic_tie_break():
    witnesses = [witness(5, 0.1), witness(2, 0.2)]
    validations = [
        validation(0, 5, 1.0, 4),
        validation(1, 2, 1.0, 3),
    ]

    selected = select_nearest_valid_witness(witnesses, validations)

    assert selected is not None
    assert selected.witness_index == 0


def test_candidate_selection_returns_none_without_valid_witnesses():
    selected = select_nearest_valid_witness(
        [witness(1, 0.1)],
        [validation(0, 1, 0.1, 5, valid=False)],
    )
    assert selected is None
