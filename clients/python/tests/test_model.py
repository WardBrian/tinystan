from pathlib import Path

import ffistan

STAN_FOLDER = Path(__file__).parent.parent.parent.parent / "test_models"


def test_model_loads():
    model = ffistan.FFIStanModel(str(STAN_FOLDER / "bernoulli" / "bernoulli.stan"))
    assert model is not None
