import json
from pathlib import Path

import numpy as np
import pytest

import ffistan

STAN_FOLDER = Path(__file__).parent.parent.parent.parent / "test_models"


def model_fixture(name):
    @pytest.fixture(scope="module", name=f"{name}_model")
    def fix():
        yield ffistan.FFIStanModel(str(STAN_FOLDER / name / f"{name}.stan"))

    return fix


bernoulli_model = model_fixture("bernoulli")
BERNOULLI_DATA = json.dumps({"N": 10, "y": [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]})

gaussian_model = model_fixture("gaussian")


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

    np.testing.assert_equal(out1["theta"], out2["theta"])

    out3 = bernoulli_model.sample(
        BERNOULLI_DATA, seed=456, num_warmup=100, num_samples=100
    )

    with pytest.raises(AssertionError):
        np.testing.assert_equal(out1["theta"], out3["theta"])


def test_save_metric(gaussian_model):
    data = json.dumps({"N": 5})
    out_unit = gaussian_model.sample(
        data,
        num_warmup=100,
        num_samples=10,
        save_metric=True,
        metric=ffistan.HMCMetric.UNIT,
    )
    assert len(out_unit.metric.shape) == 2
    assert out_unit.metric.shape[1] == 5
    np.testing.assert_equal(out_unit.metric, 1)

    out_diag = gaussian_model.sample(
        data,
        num_warmup=100,
        num_samples=10,
        save_metric=True,
        metric=ffistan.HMCMetric.DIAG,
    )
    assert len(out_diag.metric.shape) == 2
    assert out_diag.metric.shape[1] == 5
    np.testing.assert_allclose(out_diag.metric, 1)

    out_dense = gaussian_model.sample(
        data,
        num_warmup=100,
        num_samples=10,
        save_metric=True,
        metric=ffistan.HMCMetric.DENSE,
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
    assert not hasattr(out_nometric, "metric")


def test_multiple_inits():
    # well-separated mixture of gaussians
    model = ffistan.FFIStanModel(str(STAN_FOLDER / "multimodal" / "multimodal.stan"))
    init1 = {"mu": -10}
    out1 = model.sample(num_chains=2, num_warmup=100, num_samples=100, inits=init1)
    assert np.all(out1["mu"] < 0)

    init2 = {"mu": 10}
    out2 = model.sample(
        num_chains=2, num_warmup=100, num_samples=100, inits=[init1, init2]
    )
    assert np.all(out2["mu"][0] < 0)
    assert np.all(out2["mu"][1] > 0)


def test_bad_data(bernoulli_model):
    data = {"N": -1}
    with pytest.raises(RuntimeError):
        bernoulli_model.sample(data=json.dumps(data))

    data2 = {"N": 1, "y": [1, 2]}
    with pytest.raises(RuntimeError):
        bernoulli_model.sample(data=json.dumps(data2))

    with pytest.raises(RuntimeError):
        bernoulli_model.sample(data="{'bad'}")

    with pytest.raises(ValueError):
        bernoulli_model.sample(data="path/not/here.json")


def test_bad_init(bernoulli_model):
    init1 = json.dumps({"theta": 2})  # out of bounds
    with pytest.raises(RuntimeError):
        bernoulli_model.sample(BERNOULLI_DATA, inits=init1)

    with pytest.raises(ValueError):
        bernoulli_model.sample(BERNOULLI_DATA, inits="bad/path.json")

    init2 = json.dumps({"theta": 0.2})

    inits = [init2, init1]
    with pytest.raises(RuntimeError):
        bernoulli_model.sample(BERNOULLI_DATA, num_chains=2, inits=inits)

    inits = [init2, init2]
    with pytest.raises(ValueError):
        bernoulli_model.sample(BERNOULLI_DATA, num_chains=1, inits=inits)
    with pytest.raises(ValueError):
        bernoulli_model.sample(BERNOULLI_DATA, num_chains=3, inits=inits)


def test_bad_num_warmup(bernoulli_model):
    with pytest.raises(ValueError):
        bernoulli_model.sample(BERNOULLI_DATA, save_warmup=False, num_warmup=-1)
    with pytest.raises(ValueError):
        bernoulli_model.sample(BERNOULLI_DATA, save_warmup=True, num_warmup=-1)


def test_model_no_params():
    model = ffistan.FFIStanModel(str(STAN_FOLDER / "empty" / "empty.stan"))
    with pytest.raises(ValueError):
        model.sample()


@pytest.mark.parametrize(
    "arg, value",
    [
        ("num_chains", 0),
        ("num_samples", 0),
        ("num_warmup", -1),
        ("id", 0),
        ("init_radius", -0.1),
        ("delta", -0.1),
        ("delta", 1.1),
        ("gamma", 0),
        ("kappa", 0),
        ("t0", 0),
        ("stepsize", 0.0),
        ("stepsize_jitter", -0.1),
        ("stepsize_jitter", 1.1),
        ("max_depth", 0),
        ("num_threads", 0),
    ],
)
def test_bad_argument(bernoulli_model, arg, value):
    with pytest.raises(ValueError):
        bernoulli_model.sample(BERNOULLI_DATA, **{arg: value})
