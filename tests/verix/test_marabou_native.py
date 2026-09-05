import numpy as np
import pytest


maraboupy = pytest.importorskip("maraboupy")

from maraboupy import Marabou, MarabouCore


def native_options():
    return Marabou.createOptions(
        numWorkers=1,
        timeoutInSeconds=5,
        verbosity=0,
        solveWithMILP=False,
    )


def relu_query(*, output_lower_bound):
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(2)
    query.setLowerBound(0, -1.0)
    query.setUpperBound(0, -1.0)
    query.setLowerBound(1, output_lower_bound)
    query.setUpperBound(1, 1.0)
    MarabouCore.addReluConstraint(query, 0, 1)
    return query


def test_native_marabou_solver_returns_a_valid_relu_witness():
    exit_code, values, stats = MarabouCore.solve(
        relu_query(output_lower_bound=0.0),
        native_options(),
        "",
    )

    assert str(exit_code).lower() == "sat"
    assert not stats.hasTimedOut()
    assert np.isclose(values[0], -1.0)
    assert np.isclose(values[1], 0.0)


def test_native_marabou_solver_proves_an_infeasible_relu_query():
    exit_code, _, stats = MarabouCore.solve(
        relu_query(output_lower_bound=0.5),
        native_options(),
        "",
    )

    assert str(exit_code).lower() == "unsat"
    assert not stats.hasTimedOut()
