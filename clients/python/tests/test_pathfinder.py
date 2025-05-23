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
    temp_json,
)


def test_data(bernoulli_model):
    # data is a string
    out1 = bernoulli_model.pathfinder(BERNOULLI_DATA)
    assert 0.2 < out1["theta"].mean() < 0.3

    # data stored in a file
    data_file = STAN_FOLDER / "bernoulli" / "bernoulli.data.json"
    out2 = bernoulli_model.pathfinder(data=data_file)
    assert 0.2 < out2["theta"].mean() < 0.3


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
        BERNOULLI_DATA, num_paths=1, num_draws=103, num_multi_draws=104
    )
    assert out2["theta"].shape == (104,)

    out3 = bernoulli_model.pathfinder(
        BERNOULLI_DATA,
        num_paths=2,
        num_draws=105,
        num_multi_draws=1,
        calculate_lp=False,
    )

    assert out3["theta"].shape == (2 * 105,)

    out4 = bernoulli_model.pathfinder(
        BERNOULLI_DATA,
        num_paths=3,
        num_draws=107,
        num_multi_draws=1,
        psis_resample=False,
    )

    assert out4["theta"].shape == (3 * 107,)

    out5 = bernoulli_model.pathfinder(
        BERNOULLI_DATA,
        num_paths=1,
        num_draws=109,
        num_multi_draws=1,
        psis_resample=False,
    )

    assert out5["theta"].shape == (109,)

    # edge cases where num_elbo_draws > num_draws
    out6 = bernoulli_model.pathfinder(
        BERNOULLI_DATA,
        num_paths=1,
        num_multi_draws=1,
    )
    assert out6["theta"].shape == (1,)

    out7 = bernoulli_model.pathfinder(
        BERNOULLI_DATA,
        num_paths=1,
        num_draws=1,
        psis_resample=False,
    )
    assert out7["theta"].shape == (1,)

    out8 = bernoulli_model.pathfinder(
        BERNOULLI_DATA,
        num_paths=4,
        num_draws=1,
        psis_resample=False,
    )

    assert out8["theta"].shape == (4,)


def test_calculate_lp(bernoulli_model):
    out = bernoulli_model.pathfinder(BERNOULLI_DATA, num_paths=2, calculate_lp=False)
    assert np.sum(np.isnan(out["lp__"])) > 0
    assert np.sum(np.isnan(out["lp__"])) < 2000  # some calculations still needed

    out_single = bernoulli_model.pathfinder(
        BERNOULLI_DATA, num_paths=1, calculate_lp=False
    )
    assert np.sum(np.isnan(out_single["lp__"])) > 0
    assert np.sum(np.isnan(out_single["lp__"])) < 1000


def test_inits(multimodal_model, temp_json):
    # well-separated mixture of gaussians

    init1 = {"mu": -1000}
    out1 = multimodal_model.pathfinder(inits=init1)
    assert np.all(out1["mu"] < 0)

    init2 = {"mu": 1000}
    out2 = multimodal_model.pathfinder(inits=[init2])
    assert np.all(out2["mu"] > 0)

    temp_json.write(json.dumps(init1).encode())
    temp_json.flush()
    out3 = multimodal_model.pathfinder(num_paths=2, inits=[temp_json.name, init1])
    assert np.all(out3["mu"] < 0)


def test_inits_mode(multimodal_model):
    # initializing at mode means theres nowhere to go
    init1 = {"mu": -100}

    with pytest.raises(
        RuntimeError, match="None of the LBFGS iterations completed successfully"
    ):
        multimodal_model.pathfinder(inits=init1)

    # also for single path
    with pytest.raises(
        RuntimeError, match="None of the LBFGS iterations completed successfully"
    ):
        multimodal_model.pathfinder(inits=init1, num_paths=1, psis_resample=False)


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
    with pytest.raises(ValueError, match="no parameters"):
        empty_model.pathfinder()


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
