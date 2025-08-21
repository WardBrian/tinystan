import numpy as np
import pytest

import tinystan
from tests import (
    BERNOULLI_DATA,
    STAN_FOLDER,
    bernoulli_model,
    gaussian_model,
    simple_jacobian_model,
)

BERNOULLI_MODE = {"theta": 0.25}


def test_data(bernoulli_model):
    bernoulli_mode_jac = bernoulli_model.optimize(BERNOULLI_DATA, jacobian=True)
    # data is a string
    out1 = bernoulli_model.laplace_sample(bernoulli_mode_jac, BERNOULLI_DATA)
    assert 0.22 < out1["theta"].mean() < 0.28

    # data stored in a file
    data_file = STAN_FOLDER / "bernoulli" / "bernoulli.data.json"
    out2 = bernoulli_model.laplace_sample(bernoulli_mode_jac, data_file)
    assert 0.22 < out2["theta"].mean() < 0.28


def test_num_draws(bernoulli_model):
    out1 = bernoulli_model.laplace_sample(BERNOULLI_MODE, BERNOULLI_DATA, num_draws=223)
    assert out1["theta"].shape == (223,)


def test_calculate_lp(bernoulli_model):
    out1 = bernoulli_model.laplace_sample(
        BERNOULLI_MODE, BERNOULLI_DATA, num_draws=500, calculate_lp=True
    )
    assert np.sum(np.isnan(out1["log_p__"])) == 0

    out2 = bernoulli_model.laplace_sample(
        BERNOULLI_MODE, BERNOULLI_DATA, num_draws=500, calculate_lp=False
    )
    assert np.sum(np.isnan(out2["log_p__"])) == 500


@pytest.mark.parametrize("jacobian", [True, False], ids=["jacobian", "no_jacobian"])
def test_jacobian(simple_jacobian_model, jacobian):
    mode = simple_jacobian_model.optimize(jacobian=jacobian, seed=12345)

    out = simple_jacobian_model.laplace_sample(mode, jacobian=jacobian, seed=12345)

    if jacobian:
        optimum = 3.3
    else:
        optimum = 3
    assert np.isclose(out["sigma"].mean(), optimum, atol=0.2)


def test_save_hessian(gaussian_model):
    data = {"N": 3}
    mode = {"alpha": [0.1, 0.2, 0.3]}

    out = gaussian_model.laplace_sample(mode, data, save_hessian=True)
    np.testing.assert_almost_equal(
        out.hessian, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    )


def test_seed(bernoulli_model):
    out1 = bernoulli_model.laplace_sample(
        BERNOULLI_MODE,
        BERNOULLI_DATA,
        seed=123,
    )
    out2 = bernoulli_model.laplace_sample(
        BERNOULLI_MODE,
        BERNOULLI_DATA,
        seed=123,
    )

    np.testing.assert_equal(out1["theta"], out2["theta"])

    out3 = bernoulli_model.laplace_sample(
        BERNOULLI_MODE,
        BERNOULLI_DATA,
        seed=456,
    )

    with pytest.raises(AssertionError):
        np.testing.assert_equal(out1["theta"], out3["theta"])


def test_bad_data(bernoulli_model):
    bernoulli_mode_jac = bernoulli_model.optimize(BERNOULLI_DATA, jacobian=True)

    data = {"N": -1}
    with pytest.raises(RuntimeError, match="greater than or equal to 0"):
        bernoulli_model.laplace_sample(bernoulli_mode_jac, data=(data))

    data2 = {"N": 1, "y": [1, 2]}
    with pytest.raises(RuntimeError, match="mismatch in dimension"):
        bernoulli_model.laplace_sample(bernoulli_mode_jac, data=(data2))

    with pytest.raises(RuntimeError, match="Error in JSON parsing"):
        bernoulli_model.laplace_sample(bernoulli_mode_jac, data="{'bad'}")

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.laplace_sample(bernoulli_mode_jac, data="path/not/here.json")


def test_bad_mode_stanoutput(bernoulli_model):
    mode1 = bernoulli_model.sample(BERNOULLI_DATA)
    with pytest.raises(ValueError, match="can only be used with"):
        bernoulli_model.laplace_sample(mode1, BERNOULLI_DATA)


def test_bad_mode_array(bernoulli_model):
    mode1 = np.array([2.0])
    with pytest.raises(RuntimeError, match="Bounded variable is 2"):
        bernoulli_model.laplace_sample(mode1, BERNOULLI_DATA)

    mode2 = np.array([])
    with pytest.raises(ValueError, match="incorrect length"):
        bernoulli_model.laplace_sample(mode2, BERNOULLI_DATA)


def test_bad_mode_json(bernoulli_model):
    mode1 = {"theta": 2}  # out of bounds
    with pytest.raises(RuntimeError, match="Bounded variable is 2"):
        bernoulli_model.laplace_sample(mode1, BERNOULLI_DATA)

    mode2 = {"theta": [0.1, 0.1]}
    with pytest.raises(RuntimeError, match="mismatch in number"):
        bernoulli_model.laplace_sample(mode2, BERNOULLI_DATA)

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.laplace_sample("bad/path.json", BERNOULLI_DATA)


@pytest.mark.parametrize(
    "arg, value, match",
    [
        ("num_draws", 0, "at least 1"),
    ],
)
def test_bad_argument(bernoulli_model, arg, value, match):
    with pytest.raises(ValueError, match=match):
        bernoulli_model.laplace_sample(BERNOULLI_DATA, **{arg: value})
