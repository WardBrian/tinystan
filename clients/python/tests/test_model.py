from pathlib import Path

import tinystan

STAN_FOLDER = Path(__file__).parent.parent.parent.parent / "test_models"


def test_model_loads():
    model = tinystan.Model(STAN_FOLDER / "bernoulli" / "bernoulli_model.so")
    assert model is not None


def test_api_version():
    model = tinystan.Model(STAN_FOLDER / "bernoulli" / "bernoulli_model.so")
    assert model.api_version() == tinystan.__version.__version_info__


def test_stan_version():
    model = tinystan.Model(STAN_FOLDER / "bernoulli" / "bernoulli_model.so")
    stan_version = model.stan_version()
    assert stan_version[0] == 2
    assert stan_version[1] >= 34
    assert stan_version[2] >= 0
