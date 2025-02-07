import json

import numpy as np
import pytest

import tinystan
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
    out1 = bernoulli_model.sample(BERNOULLI_DATA)
    assert 0.2 < out1["theta"].mean() < 0.3

    # data stored in a file
    data_file = STAN_FOLDER / "bernoulli" / "bernoulli.data.json"
    out2 = bernoulli_model.sample(data=data_file)
    assert 0.2 < out2["theta"].mean() < 0.3


def test_save_warmup(bernoulli_model):
    out = bernoulli_model.sample(
        BERNOULLI_DATA, num_warmup=12, num_samples=34, save_warmup=False
    )
    assert out["theta"].shape[1] == 34

    out = bernoulli_model.sample(
        BERNOULLI_DATA, num_warmup=12, num_samples=34, save_warmup=True
    )
    assert out["theta"].shape[1] == 12 + 34


def test_seed(bernoulli_model):
    out1 = bernoulli_model.sample(
        BERNOULLI_DATA, seed=123, num_warmup=100, num_samples=100
    )
    out2 = bernoulli_model.sample(
        BERNOULLI_DATA, seed=123, num_warmup=100, num_samples=100
    )

    np.testing.assert_equal(out1.data, out2.data)

    out3 = bernoulli_model.sample(
        BERNOULLI_DATA, seed=456, num_warmup=100, num_samples=100
    )

    with pytest.raises(AssertionError):
        np.testing.assert_equal(out1["theta"], out3["theta"])


def test_save_metric(gaussian_model):
    data = {"N": 5}
    out_unit = gaussian_model.sample(
        data,
        num_warmup=100,
        num_samples=10,
        save_metric=True,
        metric=tinystan.HMCMetric.UNIT,
    )
    assert len(out_unit.metric.shape) == 2
    assert out_unit.metric.shape[1] == 5
    np.testing.assert_equal(out_unit.metric, 1)

    out_diag = gaussian_model.sample(
        data,
        num_warmup=100,
        num_samples=10,
        save_metric=True,
        metric=tinystan.HMCMetric.DIAGONAL,
    )
    assert len(out_diag.metric.shape) == 2
    assert out_diag.metric.shape[1] == 5
    np.testing.assert_allclose(out_diag.metric, 1)

    out_dense = gaussian_model.sample(
        data,
        num_warmup=100,
        num_samples=10,
        save_metric=True,
        metric=tinystan.HMCMetric.DENSE,
    )
    assert len(out_dense.metric.shape) == 3
    assert out_dense.metric.shape[1] == 5
    assert out_dense.metric.shape[2] == 5
    np.testing.assert_allclose(
        out_dense.metric, np.repeat(np.eye(5)[np.newaxis], 4, axis=0)
    )

    out_nometric = gaussian_model.sample(
        data, num_warmup=10, num_samples=10, save_metric=False
    )
    assert out_nometric.metric is None


def test_sundials_ode():
    sir = tinystan.Model(STAN_FOLDER / "sir" / "sir_model.so")
    sir_data = STAN_FOLDER / "sir" / "sir.data.json"
    sir_init = STAN_FOLDER / "sir" / "sir.init.json"

    out = sir.sample(
        data=sir_data,
        inits=sir_init,
        num_warmup=200,
        num_samples=100,
        num_chains=2,
        refresh=50,
    )

    assert 1.8 < out["recovery_time"].mean() < 1.9


@pytest.mark.parametrize("adapt", [True, False], ids=["adapt", "no adapt"])
def test_init_inv_metric_used(gaussian_model, adapt):
    data = {"N": 3}
    # insane metric to generate divergences
    diag_metric = np.ones((2, 3))
    diag_metric[0, :] = 1e20
    out_diag = gaussian_model.sample(
        data,
        num_chains=2,
        save_warmup=True,
        adapt=adapt,
        metric=tinystan.HMCMetric.DIAGONAL,
        init_inv_metric=diag_metric,
        save_metric=True,
        seed=1234,
    )

    # first chain should have a lot of divergences
    # second chain should have almost none
    chain_one_divergences = out_diag["divergent__"][0].sum()
    assert chain_one_divergences > (12 if adapt else 500)
    chain_two_divergences = out_diag["divergent__"][1].sum()
    assert chain_two_divergences < 12
    assert chain_two_divergences < chain_one_divergences

    # note: when adapt=false, returned metric is all 0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out_diag.metric, diag_metric)

    dense_metric = np.repeat(np.eye(3)[np.newaxis], 2, axis=0)
    dense_metric[0, np.eye(3, dtype=bool)] = 1e20
    out_dense = gaussian_model.sample(
        data,
        num_chains=2,
        save_warmup=True,
        adapt=adapt,
        metric=tinystan.HMCMetric.DENSE,
        init_inv_metric=dense_metric,
        save_metric=True,
        seed=1234,
    )

    chain_one_divergences = out_dense["divergent__"][0].sum()
    assert chain_one_divergences > (12 if adapt else 500)
    chain_two_divergences = out_dense["divergent__"][1].sum()
    assert chain_two_divergences < 12
    assert chain_two_divergences < chain_one_divergences

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out_dense.metric, dense_metric)


def test_multiple_inits(multimodal_model, temp_json):
    # well-separated mixture of gaussians
    # same init for each chain
    init1 = {"mu": -100}
    out1 = multimodal_model.sample(
        num_chains=2, num_warmup=100, num_samples=100, inits=init1
    )
    assert np.all(out1["mu"] < 0)

    # different inits for each chain
    init2 = {"mu": 100}
    out2 = multimodal_model.sample(
        num_chains=2, num_warmup=100, num_samples=100, inits=[init1, init2]
    )
    assert np.all(out2["mu"][0] < 0)
    assert np.all(out2["mu"][1] > 0)

    # mix of files and json
    temp_json.write(json.dumps(init1).encode())
    temp_json.flush()
    out3 = multimodal_model.sample(
        num_chains=2, num_warmup=100, num_samples=100, inits=[temp_json.name, init2]
    )
    assert np.all(out3["mu"][0] < 0)
    assert np.all(out3["mu"][1] > 0)


def test_init_from_other(bernoulli_model):
    out = bernoulli_model.optimize(BERNOULLI_DATA, jacobian=True)
    out2 = bernoulli_model.sample(
        BERNOULLI_DATA, num_warmup=100, num_samples=100, inits=out
    )
    assert 0.2 < out2["theta"].mean() < 0.3


def test_bad_data(bernoulli_model):
    data = {"N": -1}
    with pytest.raises(RuntimeError, match="greater than or equal to 0"):
        bernoulli_model.sample(data=(data))

    data2 = {"N": 1, "y": [1, 2]}
    with pytest.raises(RuntimeError, match="mismatch in dimension"):
        bernoulli_model.sample(data=(data2))

    with pytest.raises(RuntimeError, match="Error in JSON parsing"):
        bernoulli_model.sample(data="{'bad'}")

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.sample(data="path/not/here.json")


def test_bad_init(bernoulli_model):
    init1 = {"theta": 2}  # out of bounds
    with pytest.raises(RuntimeError, match="Initialization failed"):
        bernoulli_model.sample(BERNOULLI_DATA, inits=init1)

    with pytest.raises(ValueError, match="Could not open data file"):
        bernoulli_model.sample(BERNOULLI_DATA, inits="bad/path.json")

    init2 = {"theta": 0.2}

    # half of all inits fail
    inits = [init2, init1] * 6
    with pytest.raises(RuntimeError, match="Initialization failed"):
        bernoulli_model.sample(BERNOULLI_DATA, num_chains=12, inits=inits)

    inits = [init2, init2]
    with pytest.raises(ValueError, match="match the number of chains"):
        bernoulli_model.sample(BERNOULLI_DATA, num_chains=1, inits=inits)
    with pytest.raises(ValueError, match="match the number of chains"):
        bernoulli_model.sample(BERNOULLI_DATA, num_chains=3, inits=inits)


def test_bad_initial_metric_size(gaussian_model):
    data = {"N": 5}

    with pytest.raises(ValueError, match="Invalid initial metric size"):
        gaussian_model.sample(
            data, metric=tinystan.HMCMetric.DENSE, init_inv_metric=np.ones((5,))
        )

    with pytest.raises(ValueError, match="Invalid initial metric size"):
        gaussian_model.sample(
            data, metric=tinystan.HMCMetric.DENSE, init_inv_metric=np.ones((4, 5))
        )

    with pytest.raises(ValueError, match="Invalid initial metric size"):
        gaussian_model.sample(
            data,
            num_chains=4,
            metric=tinystan.HMCMetric.DENSE,
            init_inv_metric=np.ones((3, 5, 5)),
        )

    with pytest.raises(ValueError, match="Invalid initial metric size"):
        gaussian_model.sample(
            data, metric=tinystan.HMCMetric.DIAGONAL, init_inv_metric=np.ones((4,))
        )
    with pytest.raises(ValueError, match="Invalid initial metric size"):
        gaussian_model.sample(
            data,
            num_chains=4,
            metric=tinystan.HMCMetric.DIAGONAL,
            init_inv_metric=np.ones((3, 5)),
        )
    with pytest.raises(ValueError, match="Invalid initial metric size"):
        gaussian_model.sample(
            data,
            num_chains=4,
            metric=tinystan.HMCMetric.DIAGONAL,
            init_inv_metric=np.ones((3, 5, 5)),
        )


def test_bad_initial_metric(gaussian_model):
    data = {"N": 3}
    with pytest.raises(RuntimeError, match="not positive definite"):
        gaussian_model.sample(
            data,
            metric=tinystan.HMCMetric.DENSE,
            init_inv_metric=np.ones((3, 3)) * 1e20,
        )

    metric = np.zeros((2, 3, 3))
    metric[0, :, :] = 1e20
    metric[1, :, :] = np.eye(3)

    with pytest.raises(RuntimeError, match="not positive definite"):
        gaussian_model.sample(
            data, num_chains=2, metric=tinystan.HMCMetric.DENSE, init_inv_metric=metric
        )


def test_bad_num_warmup(bernoulli_model):
    with pytest.raises(ValueError, match="non-negative"):
        bernoulli_model.sample(BERNOULLI_DATA, save_warmup=False, num_warmup=-1)
    with pytest.raises(ValueError, match="non-negative"):
        bernoulli_model.sample(BERNOULLI_DATA, save_warmup=True, num_warmup=-1)


def test_model_no_params(empty_model):
    fit = empty_model.sample(save_metric=True)
    assert len(fit.parameters) == 7 # just HMC parameters
    assert fit.metric.size == 0

@pytest.mark.parametrize(
    "arg, value, match",
    [
        ("num_chains", 0, "at least 1"),
        ("num_samples", 0, "at least 1"),
        ("id", 0, "positive"),
        ("init_radius", -0.1, "non-negative"),
        ("delta", -0.1, "between 0 and 1"),
        ("delta", 1.1, "between 0 and 1"),
        ("gamma", 0, "positive"),
        ("kappa", 0, "positive"),
        ("t0", 0, "positive"),
        ("stepsize", 0.0, "positive"),
        ("stepsize_jitter", -0.1, "between 0 and 1"),
        ("stepsize_jitter", 1.1, "between 0 and 1"),
        ("max_depth", 0, "positive"),
        ("num_threads", 0, "positive"),
    ],
)
def test_bad_argument(bernoulli_model, arg, value, match):
    with pytest.raises(ValueError, match=match):
        bernoulli_model.sample(BERNOULLI_DATA, **{arg: value})
