import tempfile
import json

import numpy as np
import pytest

from tests import (
    BERNOULLI_DATA,
    STAN_FOLDER,
    bernoulli_model,
    empty_model,
    gaussian_model,
    multimodal_model,
)


def test_data(bernoulli_model):
    # data is a string
    out1 = bernoulli_model.pathfinder(BERNOULLI_DATA)
    assert 0.2 < out1["theta"].mean() < 0.3

    # data stored in a file
    data_file = str(STAN_FOLDER / "bernoulli" / "bernoulli.data.json")
    out2 = bernoulli_model.pathfinder(data=data_file)
    assert 0.2 < out2["theta"].mean() < 0.3


@pytest.mark.xfail(reason="stan issue #3237", strict=True)
def test_seed(bernoulli_model):
    out1 = bernoulli_model.pathfinder(BERNOULLI_DATA, seed=123)
    out2 = bernoulli_model.pathfinder(BERNOULLI_DATA, seed=123)

    np.testing.assert_equal(out1["theta"], out2["theta"])

    out3 = bernoulli_model.pathfinder(BERNOULLI_DATA, seed=456)

    with pytest.raises(AssertionError):
        np.testing.assert_equal(out1["theta"], out3["theta"])


def test_output_sizes(bernoulli_model):
    out1 = bernoulli_model.pathfinder(
        BERNOULLI_DATA, num_paths=4, num_draws=101, num_multi_draws=99
    )
    assert out1["theta"].shape == (99,)

    out2 = bernoulli_model.pathfinder(
        BERNOULLI_DATA, num_paths=1, num_draws=101, num_multi_draws=99
    )
    assert out2["theta"].shape == (101,)


def test_inits(multimodal_model):
    # well-separated mixture of gaussians

    init1 = {"mu": -1000}
    out1 = multimodal_model.pathfinder(inits=init1)
    assert np.all(out1["mu"] < 0)

    init2 = {"mu": 1000}
    out2 = multimodal_model.pathfinder(inits=[init2])
    assert np.all(out2["mu"] > 0)

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        f.write(json.dumps(init1).encode())
        f.flush()
        out3 = multimodal_model.pathfinder(num_paths=2, inits=[f.name, init1])
        assert np.all(out3["mu"] < 0)


def test_bad_data(bernoulli_model):
    data = {"N": -1}
    with pytest.raises(RuntimeError, match="greater than or equal to 0"):
        bernoulli_model.pathfinder(data=(data))

    data2 = {"N": 1, "y": [1, 2]}
    with pytest.raises(RuntimeError, match="mismatch in dimension"):
        bernoulli_model.pathfinder(data=(data2))

    with pytest.raises(RuntimeError, match="Error in JSON parsing"):
        bernoulli_model.pathfinder(data="{'bad'}")

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.pathfinder(data="path/not/here.json")


def test_bad_init(bernoulli_model):
    init1 = {"theta": 2}  # out of bounds
    with pytest.raises(RuntimeError, match="Initialization failed"):
        bernoulli_model.pathfinder(BERNOULLI_DATA, inits=init1)

    with pytest.raises(RuntimeError, match="Initialization failed"):
        bernoulli_model.pathfinder(BERNOULLI_DATA, num_paths=1, inits=init1)

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.pathfinder(BERNOULLI_DATA, inits="bad/path.json")

    inits = [init1, init1]
    with pytest.raises(RuntimeError, match="Initialization failed"):
        bernoulli_model.pathfinder(BERNOULLI_DATA, num_paths=2, inits=inits)

    init2 = {"theta": 0.2}

    # unlike sample, a failure of subset of inits is not fatal
    inits = [init1 for _ in range(10)] + [init2]
    bernoulli_model.pathfinder(
        BERNOULLI_DATA, num_paths=11, num_multi_draws=10, inits=inits
    )

    inits = [init2, init2]
    with pytest.raises(ValueError, match="match the number of chains"):
        bernoulli_model.pathfinder(BERNOULLI_DATA, num_paths=1, inits=inits)
    with pytest.raises(ValueError, match="match the number of chains"):
        bernoulli_model.pathfinder(BERNOULLI_DATA, num_paths=3, inits=inits)


def test_model_no_params(empty_model):
    out = empty_model.pathfinder()
    assert len(out.parameters) == 2  # lp, lp_approx


@pytest.mark.parametrize(
    "arg, value, match",
    [
        ("num_paths", 0, "at least 1"),
        ("num_draws", 0, "at least 1"),
        ("id", 0, "positive"),
        ("init_radius", -0.1, "non-negative"),
        ("num_threads", 0, "positive"),
        ("num_iterations", 0, "positive"),
        ("num_elbo_draws", 0, "positive"),
        ("num_multi_draws", 0, "at least 1"),
        # l-bfgs sub-options:
        ("max_history_size", 0, "positive"),
        ("init_alpha", 0, "positive"),
        ("tol_obj", 0, "positive"),
        ("tol_rel_obj", 0, "positive"),
        ("tol_grad", 0, "positive"),
        ("tol_rel_grad", 0, "positive"),
        ("tol_param", 0, "positive"),
    ],
)
def test_bad_argument(bernoulli_model, arg, value, match):
    with pytest.raises(ValueError, match=match):
        bernoulli_model.pathfinder(BERNOULLI_DATA, **{arg: value})
