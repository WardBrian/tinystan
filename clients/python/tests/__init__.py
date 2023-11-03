import json
import tempfile
from pathlib import Path

import pytest

import ffistan

STAN_FOLDER = Path(__file__).parent.parent.parent.parent / "test_models"


@pytest.fixture
def temp_json():
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    try:
        yield f
    finally:
        try:
            f.close()
            Path(f.name).unlink()
        except:
            pass


def model_fixture(name):
    @pytest.fixture(scope="module", name=f"{name}_model")
    def fix():
        yield ffistan.FFIStanModel(str(STAN_FOLDER / name / f"{name}_model.so"))

    return fix


bernoulli_model = model_fixture("bernoulli")
BERNOULLI_DATA = json.dumps({"N": 10, "y": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]})

gaussian_model = model_fixture("gaussian")

empty_model = model_fixture("empty")
multimodal_model = model_fixture("multimodal")
simple_jacobian_model = model_fixture("simple_jacobian")
