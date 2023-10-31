import os
import platform
import subprocess
import warnings
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
WINDOWS_PATH_SET = False


def verify_ffistan_path(path: str) -> None:
    folder = Path(path).resolve()
    if not folder.exists():
        raise ValueError(
            f"FFIStan folder '{folder}' does not exist!\n"
            "If you need to set a different location, call 'set_ffistan_path()'"
        )
    makefile = folder / "Makefile"
    if not makefile.exists():
        raise ValueError(
            f"FFIStan folder '{folder}' does not "
            "contain file 'Makefile', please ensure it is built properly!\n"
            "If you need to set a different location, call 'set_ffistan_path()'"
        )


def set_ffistan_path(path: str) -> None:
    """
    Set the path to FFIStan.

    This should point to the top-level folder of the repository.
    """
    path = os.path.abspath(path)
    verify_ffistan_path(path)
    os.environ["FFISTAN"] = path


def get_ffistan_path():
    """
    Get the path to FFIStan.

    By default this is set to the value of the environment
    variable ``FFISTAN``.

    If there is no path set, this function will download
    a matching version of FFIStan to a folder called
    ``.ffistan`` in the user's home directory.

    See also :func:`set_ffistan_path`
    """
    path = os.getenv("FFISTAN", "")
    if path == "":
        # TODO: download FFIStan
        raise ValueError(
            "FFIStan path not set! Please call 'set_ffistan_path()' "
            "with the location of the FFIStan repository."
        )
    return path


def windows_dll_path_setup():
    """Add tbb.dll to %PATH% on Windows."""
    global WINDOWS_PATH_SET
    if IS_WINDOWS and not WINDOWS_PATH_SET:
        try:
            out = subprocess.run(
                ["where.exe", "tbb.dll"], check=True, capture_output=True
            )
            tbb_path = os.path.dirname(out.stdout.decode().splitlines()[0])
            os.add_dll_directory(tbb_path)
        except:
            try:
                tbb_path = os.path.abspath(
                    os.path.join(
                        get_ffistan_path(), "stan", "lib", "stan_math", "lib", "tbb"
                    )
                )
                os.environ["PATH"] = tbb_path + ";" + os.environ["PATH"]
                os.add_dll_directory(tbb_path)
                WINDOWS_PATH_SET = True
            except:
                warnings.warn(
                    "Unable to set path to TBB's DLL. Loading FFIStan models may fail. "
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
            WINDOWS_PATH_SET &= True
        except:
            # no default location
            warnings.warn(
                "Unable to find MinGW's DLL location. Loading FFIStan models may fail.",
                RuntimeWarning,
            )
            WINDOWS_PATH_SET = False
