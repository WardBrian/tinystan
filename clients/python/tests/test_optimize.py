import json

import numpy as np
import pytest

import ffistan
from tests import (
    BERNOULLI_DATA,
    STAN_FOLDER,
    bernoulli_model,
    empty_model,
    gaussian_model,
    multimodal_model,
    simple_jacobian_model,
    temp_json,
)

ALL_ALGORITHMS = [
    ffistan.OptimizationAlgorithm.NEWTON,
    ffistan.OptimizationAlgorithm.BFGS,
    ffistan.OptimizationAlgorithm.LBFGS,
]


def test_data(bernoulli_model):
    # data is a string
    out1 = bernoulli_model.optimize(BERNOULLI_DATA)
    assert 0.19 < out1["theta"] < 0.21

    # data stored in a file
    data_file = STAN_FOLDER / "bernoulli" / "bernoulli.data.json"
    out2 = bernoulli_model.optimize(data=data_file)
    assert 0.19 < out2["theta"] < 0.21


@pytest.mark.parametrize("jacobian", [True, False], ids=["jacobian", "no_jacobian"])
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_jacobian_algorithm(simple_jacobian_model, algorithm, jacobian):
    out = simple_jacobian_model.optimize(
        algorithm=algorithm, jacobian=jacobian, seed=1234
    )
    if jacobian:
        optimum = 3.3
    else:
        optimum = 3
    assert np.isclose(out["sigma"], optimum, atol=0.01)


def test_seed(bernoulli_model):
    out1 = bernoulli_model.optimize(
        BERNOULLI_DATA,
        seed=123,
    )
    out2 = bernoulli_model.optimize(
        BERNOULLI_DATA,
        seed=123,
    )

    np.testing.assert_equal(out1["theta"], out2["theta"])

    out3 = bernoulli_model.optimize(
        BERNOULLI_DATA,
        seed=456,
    )

    with pytest.raises(AssertionError):
        np.testing.assert_equal(out1["theta"], out3["theta"])


def test_inits(multimodal_model, temp_json):
    # well-separated mixture of gaussians
    init1 = {"mu": -1000}
    out1 = multimodal_model.optimize(init=init1)
    assert np.all(out1["mu"] < 0)

    init2 = {"mu": 1000}
    temp_json.write(json.dumps(init2).encode())
    temp_json.flush()
    out2 = multimodal_model.optimize(init=temp_json.name)
    assert np.all(out2["mu"] > 0)


def test_bad_data(bernoulli_model):
    data = {"N": -1}
    with pytest.raises(RuntimeError, match="greater than or equal to 0"):
        bernoulli_model.optimize(data=(data))

    data2 = {"N": 1, "y": [1, 2]}
    with pytest.raises(RuntimeError, match="mismatch in dimension"):
        bernoulli_model.optimize(data=(data2))

    with pytest.raises(RuntimeError, match="Error in JSON parsing"):
        bernoulli_model.optimize(data="{'bad'}")

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.optimize(data="path/not/here.json")


def test_bad_init(bernoulli_model):
    init1 = {"theta": 2}  # out of bounds
    with pytest.raises(RuntimeError, match="Initialization failed"):
        bernoulli_model.optimize(BERNOULLI_DATA, init=init1)

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.optimize(BERNOULLI_DATA, init="bad/path.json")


def test_model_no_params(empty_model):
    out = empty_model.optimize()
    assert len(out.parameters) == 1  # lp


@pytest.mark.parametrize(
    "arg, value, match",
    [
        ("id", 0, "positive"),
        ("num_iterations", 0, "positive"),
        ("init_radius", -0.1, "non-negative"),
    ],
)
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_bad_argument(bernoulli_model, algorithm, arg, value, match):
    with pytest.raises(ValueError, match=match):
        bernoulli_model.optimize(BERNOULLI_DATA, algorithm=algorithm, **{arg: value})


@pytest.mark.parametrize(
    "arg, value, match",
    [
        ("max_history_size", 0, "positive"),
    ],
)
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_bad_argument_lbfgs(bernoulli_model, algorithm, arg, value, match):
    if algorithm == ffistan.OptimizationAlgorithm.LBFGS:
        with pytest.raises(ValueError, match=match):
            bernoulli_model.optimize(
                BERNOULLI_DATA, algorithm=algorithm, **{arg: value}
            )
    else:
        bernoulli_model.optimize(BERNOULLI_DATA, algorithm=algorithm, **{arg: value})


@pytest.mark.parametrize(
    "arg, value, match",
    [
        ("init_alpha", 0, "positive"),
        ("tol_obj", 0, "positive"),
        ("tol_rel_obj", 0, "positive"),
        ("tol_grad", 0, "positive"),
        ("tol_rel_grad", 0, "positive"),
        ("tol_param", 0, "positive"),
    ],
)
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_bad_argument_bfgs(bernoulli_model, algorithm, arg, value, match):
    if algorithm != ffistan.OptimizationAlgorithm.NEWTON:
        with pytest.raises(ValueError, match=match):
            bernoulli_model.optimize(
                BERNOULLI_DATA, algorithm=algorithm, **{arg: value}
            )
    else:
        bernoulli_model.optimize(BERNOULLI_DATA, algorithm=algorithm, **{arg: value})
