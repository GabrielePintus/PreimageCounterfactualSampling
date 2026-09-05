from types import SimpleNamespace

import numpy as np

from verix import CheckRequest, CheckStatus
from verix.backends import MarabouClassificationChecker, MarabouOptions


class FakeMarabouNetwork:
    def __init__(self, solve_results):
        self.solve_results = list(solve_results)
        self.bounds = {}
        self.inequalities = []
        self.additionalEquList = []
        self.clear_property_calls = 0

    def setLowerBound(self, variable, value):
        self.bounds.setdefault(variable, {})["lower"] = value

    def setUpperBound(self, variable, value):
        self.bounds.setdefault(variable, {})["upper"] = value

    def addInequality(self, variables, coefficients, scalar, isProperty):
        self.inequalities.append(
            (tuple(variables), tuple(coefficients), scalar, isProperty)
        )

    def solve(self, options, verbose):
        return self.solve_results.pop(0)

    def clearProperty(self):
        self.clear_property_calls += 1


def make_checker(network):
    checker = object.__new__(MarabouClassificationChecker)
    checker.network = network
    checker.input_variables = np.array([10, 11])
    checker.output_variables = np.array([20, 21])
    checker.lower_bounds = np.array([0.0, 0.0])
    checker.upper_bounds = np.array([1.0, 1.0])
    checker.options = MarabouOptions()
    checker.solver_options = object()
    return checker


def make_request():
    return CheckRequest(
        x=np.array([0.2, 0.8]),
        reference_output=0,
        free_features=(0,),
        free_input_indices=(0,),
        epsilon=0.5,
        norm=np.inf,
        discrepancy=0.0,
    )


def test_marabou_uses_the_native_solver_by_default():
    assert MarabouOptions().solve_with_milp is False


def test_marabou_unsat_proves_invariance_and_sets_official_bounds():
    network = FakeMarabouNetwork(
        [("unsat", {}, SimpleNamespace(name="stats"))]
    )
    result = make_checker(network).check(make_request())

    assert result.status is CheckStatus.HOLDS
    assert network.bounds == {
        10: {"lower": 0.0, "upper": 0.7},
        11: {"lower": 0.8, "upper": 0.8},
    }
    assert network.inequalities == [
        ((20, 21), (1.0, -1.0), -1.0e-6, True)
    ]
    assert network.clear_property_calls == 1


def test_marabou_sat_returns_the_native_counterexample():
    values = {10: 0.4, 11: 0.8, 20: -0.2, 21: 0.3}
    network = FakeMarabouNetwork(
        [("sat", values, SimpleNamespace(name="stats"))]
    )
    result = make_checker(network).check(make_request())

    assert result.status is CheckStatus.VIOLATED
    assert np.array_equal(result.counterexample, np.array([0.4, 0.8]))
    assert result.counterexample_output == 1
    assert result.metadata["checked_classes"] == (1,)
    assert network.clear_property_calls == 1


def test_marabou_timeout_is_unknown_and_remains_sound():
    network = FakeMarabouNetwork(
        [("TIMEOUT", {}, SimpleNamespace(name="stats"))]
    )
    result = make_checker(network).check(make_request())

    assert result.status is CheckStatus.UNKNOWN
    assert result.counterexample is None
    assert network.clear_property_calls == 1


def test_marabou_prediction_restores_the_complete_onnx_input_shape():
    class FakeSession:
        def __init__(self):
            self.received = None

        def run(self, output_names, inputs):
            self.received = inputs["input"]
            return [np.array([[0.1, 0.9]])]

    checker = make_checker(FakeMarabouNetwork([]))
    checker.input_shape = (1, 2)
    checker.input_name = "input"
    checker.onnx_session = FakeSession()

    assert checker.predict(np.array([0.2, 0.8])) == 1
    assert checker.onnx_session.received.shape == (1, 2)
