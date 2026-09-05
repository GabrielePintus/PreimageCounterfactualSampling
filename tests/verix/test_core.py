import numpy as np
import pytest

from verix import CheckResult, CheckStatus, VeriX


class ScriptedChecker:
    is_complete = False

    def __init__(self):
        self.requests = []

    def predict(self, x):
        return 0

    def check(self, request):
        self.requests.append(request)
        if request.free_features == (2,):
            return CheckResult(CheckStatus.HOLDS)
        if request.free_features == (2, 0):
            return CheckResult(
                CheckStatus.VIOLATED,
                counterexample=np.array([1.0, 0.0, 0.25]),
                counterexample_output=1,
                metadata={"solver": "fake"},
            )
        if request.free_features == (2, 1):
            return CheckResult(CheckStatus.UNKNOWN, metadata={"timeout": True})
        raise AssertionError(f"unexpected free set {request.free_features}")


def test_algorithm_one_state_transitions_and_witnesses():
    checker = ScriptedChecker()
    result = VeriX(
        checker,
        epsilon=0.25,
        norm=np.inf,
    ).explain(
        np.zeros(3),
        traversal_order=[2, 0, 1],
    )

    assert result.reference_output == 0
    assert result.irrelevant == (2,)
    assert result.explanation == (0, 1)
    assert result.unknown == (1,)
    assert result.traversal_order == (2, 0, 1)
    assert not result.locally_minimal

    assert [request.free_features for request in checker.requests] == [
        (2,),
        (2, 0),
        (2, 1),
    ]
    assert len(result.counterfactuals) == 1
    witness = result.counterfactuals[0]
    assert witness.feature == 0
    assert witness.free_features == (2, 0)
    assert witness.output == 1
    assert np.array_equal(witness.x, np.array([1.0, 0.0, 0.25]))
    assert result.steps[1].metadata == {"solver": "fake"}


def test_feature_groups_map_semantic_features_to_input_coordinates():
    class GroupChecker:
        is_complete = True

        def __init__(self):
            self.requests = []

        def predict(self, x):
            return 1

        def check(self, request):
            self.requests.append(request)
            return CheckResult(CheckStatus.HOLDS)

    checker = GroupChecker()
    result = VeriX(
        checker,
        epsilon=0.1,
        feature_groups=[(0,), (1, 2, 3)],
    ).explain(
        np.zeros(4),
        traversal_order=[1, 0],
    )

    assert result.feature_groups == ((0,), (1, 2, 3))
    assert result.irrelevant == (1, 0)
    assert result.explanation == ()
    assert result.locally_minimal
    assert checker.requests[0].free_input_indices == (1, 2, 3)
    assert checker.requests[1].free_input_indices == (1, 2, 3, 0)


def test_violated_check_requires_concrete_counterexample():
    class InvalidChecker:
        is_complete = True

        def predict(self, x):
            return 0

        def check(self, request):
            return CheckResult(CheckStatus.VIOLATED)

    with pytest.raises(ValueError, match="concrete counterexample"):
        VeriX(InvalidChecker(), epsilon=0.1).explain(np.zeros(2))


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [()],
        [(0,), (0, 1)],
        [(0,)],
        [(0,), (2,)],
    ],
)
def test_feature_groups_must_be_an_exact_partition(groups):
    checker = ScriptedChecker()
    with pytest.raises(ValueError, match="feature_groups"):
        VeriX(
            checker,
            epsilon=0.1,
            feature_groups=groups,
        ).explain(np.zeros(2))


def test_traversal_order_must_be_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        VeriX(ScriptedChecker(), epsilon=0.1).explain(
            np.zeros(3),
            traversal_order=[0, 0, 1],
        )


@pytest.mark.parametrize("norm", [0, 3, -np.inf])
def test_only_paper_norms_are_accepted(norm):
    with pytest.raises(ValueError, match="norm"):
        VeriX(ScriptedChecker(), epsilon=0.1, norm=norm)
