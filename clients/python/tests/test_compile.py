import os
from pathlib import Path
from unittest import mock

import pytest

import ffistan

STAN_FOLDER = Path(__file__).parent.parent.parent.parent / "test_models"


@mock.patch.dict(os.environ, {"FFISTAN": ""})
def test_download_ffistan():
    ffistan.compile.get_ffistan_path(ffistan.compile.get_ffistan_path())


def test_compile_good():
    stanfile = STAN_FOLDER / "gaussian" / "gaussian.stan"
    lib = ffistan.compile.generate_so_name(stanfile)
    lib.unlink(missing_ok=True)
    res = ffistan.compile_model(stanfile, stanc_args=["--O1"])
    assert lib.samefile(res)
    lib.unlink()

    ffistan.Model(stanfile)
    assert lib.exists()


def test_compile_bad_ext():
    not_stanfile = STAN_FOLDER / "bernoulli" / "bernoulli.data.json"
    with pytest.raises(ValueError, match=r".stan"):
        ffistan.compile_model(not_stanfile)


def test_compile_nonexistant():
    not_stanfile = STAN_FOLDER / "gaussian" / "gaussian-nothere.stan"
    with pytest.raises(FileNotFoundError):
        ffistan.compile_model(not_stanfile)


def test_compile_syntax_error():
    stanfile = STAN_FOLDER / "syntax_error" / "syntax_error.stan"
    with pytest.raises(RuntimeError, match=r"Syntax error"):
        ffistan.compile_model(stanfile)


def test_compile_bad_ffistan():
    with pytest.raises(ValueError, match=r"does not exist"):
        ffistan.compile.set_ffistan_path("dummy")
    with pytest.raises(ValueError, match=r"does not contain file 'Makefile'"):
        ffistan.compile.set_ffistan_path(STAN_FOLDER)
