import os
import platform
import subprocess
import warnings
from pathlib import Path
from typing import List, Union

from .__version import __version__
from .download import CURRENT_TINYSTAN, HOME_TINYSTAN, get_tinystan_src
from .util import validate_readable

IS_WINDOWS = platform.system() == "Windows"
MAKE = os.getenv("MAKE", "make")
WINDOWS_PATH_SET = False


def verify_tinystan_path(path: str) -> None:
    folder = Path(path).resolve()
    if not folder.exists():
        raise ValueError(
            f"TinyStan folder '{folder}' does not exist!\n"
            "If you need to set a different location, call 'set_tinystan_path()'"
        )
    makefile = folder / "Makefile"
    if not makefile.exists():
        raise ValueError(
            f"TinyStan folder '{folder}' does not "
            "contain file 'Makefile', please ensure it is built properly!\n"
            "If you need to set a different location, call 'set_tinystan_path()'"
        )


def set_tinystan_path(path: str) -> None:
    """
    Set the path to TinyStan.


    This is useful for development or if the automatic download is
    not desired. If the path is invalid, this function will raise an
    error.

    This should point to the top-level folder of the repository.
    """
    path = os.path.abspath(path)
    verify_tinystan_path(path)
    os.environ["TINYSTAN"] = path


def get_tinystan_path() -> str:
    """
    Get the path to TinyStan.

    By default this is set to the value of the environment
    variable ``TINYSTAN``.

    If there is no path set, this function will download
    a matching version of TinyStan to a folder called
    ``.tinystan`` in the user's home directory.

    See also :func:`set_tinystan_path`
    """
    path = os.getenv("TINYSTAN", "")
    if path == "":
        try:
            path = os.fspath(CURRENT_TINYSTAN)
            verify_tinystan_path(path)
        except ValueError:
            print(
                "TinyStan not found at location specified by $TINYSTAN "
                f"environment variable, downloading version {__version__} to {path}"
            )
            get_tinystan_src()
            num_files = len(list(HOME_TINYSTAN.iterdir()))
            if num_files >= 5:
                warnings.warn(
                    f"Found {num_files} different versions of TinyStan in {HOME_TINYSTAN}. "
                    "Consider deleting old versions to save space."
                )
            print("Done!")

    return path


def generate_so_name(model: Path) -> Path:
    name = model.stem
    return model.with_stem(f"{name}_model").with_suffix(".so")


def compile_model(
    stan_file: Union[str, os.PathLike],
    *,
    stanc_args: List[str] = [],
    make_args: List[str] = [],
) -> Path:
    """
    Run TinyStan's Makefile on a ``.stan`` file, creating the ``.so``
    used by the Model class.

    This function requires the presence of the TinyStan source code.
    It will download the source code if it is not found.
    A manual location can be used instead by calling :func:`set_tinystan_path`.

    Parameters
    ----------
    stan_file : Union[str, os.PathLike]
        A path to a Stan model file.
    stanc_args : List[str], optional
        A list of arguments to pass to stanc3.
        For example, ``["--O1"]`` will enable compiler optimization level 1.
    make_args : List[str], optional
        A list of additional arguments to pass to Make.
        If the same flags are defined in :file:`make/local`, the versions
        passed here will take precedent.

    Raises
    ------
    FileNotFoundError or PermissionError
        If ``stan_file`` does not exist or is not readable.
    ValueError
        If the TinyStan source code location is invalid.
    RuntimeError
        If compilation fails.

    Returns
    -------
    Path
        The path to the compiled ``.so`` file.
    """
    validate_readable(stan_file)
    verify_tinystan_path(get_tinystan_path())
    file_path = Path(stan_file).resolve()

    if file_path.suffix != ".stan":
        raise ValueError(f"File '{stan_file}' does not end in .stan")

    output = generate_so_name(file_path)
    cmd = (
        [MAKE]
        + make_args
        + ["STANCFLAGS=" + " ".join(["--include-paths=."] + stanc_args)]
        + [os.fspath(output)]
    )
    proc = subprocess.run(
        cmd, cwd=get_tinystan_path(), capture_output=True, text=True, check=False
    )

    if proc.returncode:
        error = (
            f"Command {' '.join(cmd)} failed with code {proc.returncode}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

        raise RuntimeError(error)
    return os.fspath(output)


def windows_dll_path_setup() -> None:
    """Add tbb.dll to %PATH% on Windows."""
    global WINDOWS_PATH_SET
    if IS_WINDOWS and not WINDOWS_PATH_SET:
        try:
            out = subprocess.run(
                ["where.exe", "tbb.dll"], check=True, capture_output=True
            )
            tbb_path = os.path.dirname(out.stdout.decode().splitlines()[0])
            os.add_dll_directory(tbb_path)
        except Exception:
            try:
                tbb_path = os.path.abspath(
                    os.path.join(
                        get_tinystan_path(), "stan", "lib", "stan_math", "lib", "tbb"
                    )
                )
                os.environ["PATH"] = tbb_path + ";" + os.environ["PATH"]
                os.add_dll_directory(tbb_path)
                WINDOWS_PATH_SET = True
            except Exception:
                warnings.warn(
                    "Unable to set path to TBB's DLL. Loading TinyStan models may fail. "
                    f"Tried path '{tbb_path}'",
                    RuntimeWarning,
                )
                WINDOWS_PATH_SET = False
        try:
            out = subprocess.run(
                [
                    "where.exe",
                    "libwinpthread-1.dll",
                    "libgcc_s_seh-1.dll",
                    "libstdc++-6.dll",
                ],
                check=True,
                capture_output=True,
            )
            mingw_dir = os.path.abspath(
                os.path.dirname(out.stdout.decode().splitlines()[0])
            )
            os.add_dll_directory(mingw_dir)
        except Exception:
            # no default location
            warnings.warn(
                "Unable to find MinGW's DLL location. Loading TinyStan models may fail.",
                RuntimeWarning,
            )
            WINDOWS_PATH_SET = False
