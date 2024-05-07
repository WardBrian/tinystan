import os
from pathlib import Path
from unittest import mock

import pytest

import tinystan

STAN_FOLDER = Path(__file__).parent.parent.parent.parent / "test_models"


@mock.patch.dict(os.environ, {"TINYSTAN": ""})
def test_download_tinystan():
    tinystan.compile.verify_tinystan_path(tinystan.compile.get_tinystan_path())


def test_compile_good():
    stanfile = STAN_FOLDER / "gaussian" / "gaussian.stan"
    lib = tinystan.compile.generate_so_name(stanfile)
    lib.unlink(missing_ok=True)
    res = tinystan.compile_model(stanfile, stanc_args=["--O1"])
    assert lib.samefile(res)
    lib.unlink()

    tinystan.Model(stanfile)
    assert lib.exists()


def test_compile_user_header():
    stanfile = STAN_FOLDER / "external" / "external.stan"
    lib = tinystan.compile.generate_so_name(stanfile)
    lib.unlink(missing_ok=True)

    with pytest.raises(RuntimeError, match=r"declared without specifying a definition"):
        tinystan.compile_model(stanfile)

    with pytest.raises(RuntimeError, match=r"USER_HEADER"):
        tinystan.compile_model(stanfile, stanc_args=["--allow-undefined"])

    header = stanfile.parent / "make_odds.hpp"
    res = tinystan.compile_model(
        stanfile, stanc_args=["--allow-undefined"], make_args=[f"USER_HEADER={header}"]
    )
    assert lib.samefile(res)
    assert lib.exists()


def test_compile_bad_ext():
    not_stanfile = STAN_FOLDER / "bernoulli" / "bernoulli.data.json"
    with pytest.raises(ValueError, match=r".stan"):
        tinystan.compile_model(not_stanfile)


def test_compile_nonexistant():
    not_stanfile = STAN_FOLDER / "gaussian" / "gaussian-nothere.stan"
    with pytest.raises(FileNotFoundError):
        tinystan.compile_model(not_stanfile)


def test_compile_syntax_error():
    stanfile = STAN_FOLDER / "syntax_error" / "syntax_error.stan"
    with pytest.raises(RuntimeError, match=r"Syntax error"):
        tinystan.compile_model(stanfile)


def test_compile_bad_tinystan():
    with pytest.raises(ValueError, match=r"does not exist"):
        tinystan.compile.set_tinystan_path("dummy")
    with pytest.raises(ValueError, match=r"does not contain file 'Makefile'"):
        tinystan.compile.set_tinystan_path(STAN_FOLDER)
