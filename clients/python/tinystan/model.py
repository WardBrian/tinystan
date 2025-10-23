import contextlib
import ctypes
import sys
import warnings
from enum import Enum
from os import PathLike, fspath
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import dllist
import numpy as np
from numpy.ctypeslib import ndpointer
from stanio import dump_stan_json

from .__version import __version_info__
from .compile import compile_model, windows_dll_path_setup
from .output import StanOutput
from .util import validate_readable

# type aliases
StanData = Union[str, PathLike, Mapping[str, Any]]


# ctypes helpers
def wrapped_ndptr(*args, **kwargs):
    """
    A version of np.ctypeslib.ndpointer
    which allows None (passed as NULL)
    """
    base = ndpointer(*args, **kwargs)

    def from_param(_cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)

    return type(base.__name__, (base,), {"from_param": classmethod(from_param)})


double_array = ndpointer(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
nullable_double_array = wrapped_ndptr(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
err_ptr = ctypes.POINTER(ctypes.c_void_p)
print_callback_type = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_bool
)


@print_callback_type
def print_callback(msg, size, is_error):
    print(
        ctypes.string_at(msg, size).decode("utf-8"),
        file=sys.stderr if is_error else sys.stdout,
    )


# algorithm-specific constants

HMC_SAMPLER_VARIABLES = [
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
]

PATHFINDER_VARIABLES = ["lp_approx__", "lp__", "path__"]

OPTIMIZE_VARIABLES = [
    "lp__",
]

LAPLACE_VARIABLES = [
    "log_p__",
    "log_q__",
]

FIXED_SAMPLER_VARIABLES = [
    "lp__",
    "accept_stat__",
]


class HMCMetric(Enum):
    """Choices for the structure of the inverse mass matrix used in the HMC sampler."""

    UNIT = 0  #: :meta hide-value:
    DENSE = 1  #: :meta hide-value:
    DIAGONAL = 2  #: :meta hide-value:


class OptimizationAlgorithm(Enum):
    """Choices for the optimization algorithm to use."""

    NEWTON = 0  #: :meta hide-value:
    BFGS = 1  #: :meta hide-value:
    LBFGS = 2  #: :meta hide-value:


_exception_types = [RuntimeError, ValueError, KeyboardInterrupt]


# TODO also allow inits from a StanOutput?
def encode_stan_json(data: Union[str, PathLike, Mapping[str, Any]]) -> bytes:
    """Turn the provided data into something we can send to C++."""
    if isinstance(data, PathLike):
        validate_readable(data)
        return fspath(data).encode()
    if isinstance(data, str):
        return data.encode()
    return dump_stan_json(data).encode()


def rand_u32():
    """Generate a random 32-bit unsigned integer."""
    return np.random.randint(0, 2**32 - 1, dtype=np.uint32)


def preprocess_laplace_inputs(
    mode: Union[StanOutput, np.ndarray, StanData],
) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
    if isinstance(mode, StanOutput):
        # handle case of passing optimization output directly
        if len(mode.data.shape) == 1:
            mode = mode.data[1:]
        else:
            raise ValueError("Laplace can only be used with Optimization output")
            # mode = mode.create_inits(chains=1, seed=seed)

    if isinstance(mode, np.ndarray):
        mode_json = None
        mode_array = mode
    else:
        mode_json = encode_stan_json(mode)
        mode_array = None

    return mode_array, mode_json


class Model:
    def __init__(
        self,
        model: Union[str, PathLike],
        *,
        capture_stan_prints: bool = True,
        stanc_args: List[str] = [],
        make_args: List[str] = [],
        warn: bool = True,
    ):
        """
        Load a Stan model for inference, compiling it if necessary.

        Parameters
        ----------
        model : Union[str, PathLike]
            Path to the Stan model file or shared object.
        stanc_args : List[str], optional
            A list of arguments to pass to stanc3 if the model is not compiled.
            For example, ``["--O1"]`` will enable compiler optimization level 1.
        make_args : List[str], optional
            A list of additional arguments to pass to GNU Make if the
            model is not compiled. For example, ``["STAN_NO_RANGE_CHECKS=True"]``
            will disable bounds checking in the Stan Math library. If the
            same flags are defined in ``make/local``, the versions passed here
            will take precedent.
        capture_stan_prints : bool, optional
            If ``True``, capture all ``print`` statements and output
            from Stan and print them from Python. This may have
            a performance impact. If ``False``, ``print`` statements
            from Stan will be sent to ``cout`` and will not be seen in
            Jupyter or capturable with :func:`contextlib.redirect_stdout`.
        warn : bool, optional
            If ``False``, the warning about re-loading the same shared object
            is suppressed.
        """
        windows_dll_path_setup()

        model = fspath(model)
        if model.endswith(".stan"):
            self.lib_path = fspath(
                compile_model(model, stanc_args=stanc_args, make_args=make_args)
            )
        else:
            self.lib_path = model

        self.capture_stan_prints = capture_stan_prints
        if warn and hasattr(dllist, "dllist") and self.lib_path in dllist.dllist():
            warnings.warn(
                f"Loading a shared object {self.lib_path} that has already been loaded.\n"
                "If the file has changed since the last time it was loaded, this load may "
                "not update the library!"
            )

        self._lib = ctypes.CDLL(self.lib_path)

        self._version = self._lib.tinystan_api_version
        self._version.restype = None
        self._version.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        api_ver = self.api_version()
        if api_ver[0] != __version_info__[0]:
            raise RuntimeError(
                "Incompatible TinyStan API version. Expected "
                f"{__version_info__} but got {api_ver}.\n"
                "You need to re-compile your model."
            )
        if api_ver != __version_info__:
            warnings.warn(
                "TinyStan API version does not match. Expected "
                f"{__version_info__} but got {api_ver}.\n"
                "You may need to re-compile your model."
            )

        self._create_model = self._lib.tinystan_create_model
        self._create_model.restype = ctypes.c_void_p
        self._create_model.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint,
            print_callback_type,
            err_ptr,
        ]

        self._delete_model = self._lib.tinystan_destroy_model
        self._delete_model.restype = None
        self._delete_model.argtypes = [ctypes.c_void_p]

        self._get_param_names = self._lib.tinystan_model_param_names
        self._get_param_names.restype = ctypes.c_char_p
        self._get_param_names.argtypes = [ctypes.c_void_p]

        self._num_free_params = self._lib.tinystan_model_num_free_params
        self._num_free_params.restype = ctypes.c_size_t
        self._num_free_params.argtypes = [ctypes.c_void_p]

        self._num_req_constrained_params = (
            self._lib.tinystan_model_num_constrained_params_for_unconstraining
        )
        self._num_req_constrained_params.restype = ctypes.c_size_t
        self._num_req_constrained_params.argtypes = [ctypes.c_void_p]

        self._stan_version = self._lib.tinystan_stan_version
        self._stan_version.restype = None
        self._stan_version.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        self._ffi_sample = self._lib.tinystan_sample
        self._ffi_sample.restype = ctypes.c_int
        self._ffi_sample.argtypes = [
            ctypes.c_void_p,  # model
            ctypes.c_size_t,  # num_chains
            ctypes.c_char_p,  # inits
            ctypes.c_uint,  # seed
            ctypes.c_uint,  # id
            ctypes.c_double,  # init_radius
            ctypes.c_int,  # num_warmup
            ctypes.c_int,  # num_samples
            ctypes.c_int,  # really enum for metric
            nullable_double_array,  # metric init in
            # adaptation
            ctypes.c_bool,  # adapt
            ctypes.c_double,  # delta
            ctypes.c_double,  # gamma
            ctypes.c_double,  # kappa
            ctypes.c_double,  # t0
            ctypes.c_uint,  # init_buffer
            ctypes.c_uint,  # term_buffer
            ctypes.c_uint,  # window
            ctypes.c_bool,  # save_warmup
            ctypes.c_double,  # stepsize
            ctypes.c_double,  # stepsize_jitter
            ctypes.c_int,  # max_depth
            ctypes.c_int,  # refresh
            ctypes.c_int,  # num_threads
            double_array,
            ctypes.c_size_t,  # buffer size
            nullable_double_array,  # stepsize out
            nullable_double_array,  # metric out
            err_ptr,
        ]

        self._ffi_pathfinder = self._lib.tinystan_pathfinder
        self._ffi_pathfinder.restype = ctypes.c_int
        self._ffi_pathfinder.argtypes = [
            ctypes.c_void_p,  # model
            ctypes.c_size_t,  # num_paths
            ctypes.c_char_p,  # inits
            ctypes.c_uint,  # seed
            ctypes.c_uint,  # id
            ctypes.c_double,  # init_radius
            ctypes.c_int,  # num_draws
            ctypes.c_int,  # max_history_size
            ctypes.c_double,  # init_alpha
            ctypes.c_double,  # tol_obj
            ctypes.c_double,  # tol_rel_obj
            ctypes.c_double,  # tol_grad
            ctypes.c_double,  # tol_rel_grad
            ctypes.c_double,  # tol_param
            ctypes.c_int,  # num_iterations
            ctypes.c_int,  # num_elbo_draws
            ctypes.c_int,  # num_multi_draws
            ctypes.c_bool,  # calculate_lp
            ctypes.c_bool,  # psis_resample
            ctypes.c_int,  # refresh
            ctypes.c_int,  # num_threads
            double_array,  # output samples
            ctypes.c_size_t,  # buffer size
            err_ptr,
        ]

        self._ffi_optimize = self._lib.tinystan_optimize
        self._ffi_optimize.restype = ctypes.c_int
        self._ffi_optimize.argtypes = [
            ctypes.c_void_p,  # model
            ctypes.c_char_p,  # inits
            ctypes.c_uint,  # seed
            ctypes.c_uint,  # id
            ctypes.c_double,  # init_radius
            ctypes.c_int,  # really enum for algorithm
            ctypes.c_int,  # num_iterations
            ctypes.c_bool,  # jacobian
            ctypes.c_int,  # max_history_size
            ctypes.c_double,  # init_alpha
            ctypes.c_double,  # tol_obj
            ctypes.c_double,  # tol_rel_obj
            ctypes.c_double,  # tol_grad
            ctypes.c_double,  # tol_rel_grad
            ctypes.c_double,  # tol_param
            ctypes.c_int,  # refresh
            ctypes.c_int,  # num_threads
            double_array,
            ctypes.c_size_t,  # buffer size
            err_ptr,
        ]

        self._ffi_laplace = self._lib.tinystan_laplace_sample
        self._ffi_laplace.restype = ctypes.c_int
        self._ffi_laplace.argtypes = [
            ctypes.c_void_p,  # model
            nullable_double_array,  # array of constrained params
            ctypes.c_char_p,  # json of constrained params
            ctypes.c_uint,  # seed
            ctypes.c_int,  # draws
            ctypes.c_bool,  # jacobian
            ctypes.c_bool,  # calculate_lp
            ctypes.c_int,  # refresh
            ctypes.c_int,  # num_threads
            double_array,  # draws buffer
            ctypes.c_size_t,  # buffer size
            nullable_double_array,  # hessian out
            err_ptr,
        ]

        self._get_error_msg = self._lib.tinystan_get_error_message
        self._get_error_msg.restype = ctypes.c_char_p
        self._get_error_msg.argtypes = [ctypes.c_void_p]
        self._get_error_type = self._lib.tinystan_get_error_type
        self._get_error_type.restype = ctypes.c_int  # really enum
        self._get_error_type.argtypes = [ctypes.c_void_p]
        self._free_error = self._lib.tinystan_destroy_error
        self._free_error.restype = None
        self._free_error.argtypes = [ctypes.c_void_p]

        get_separator = self._lib.tinystan_separator_char
        get_separator.restype = ctypes.c_char
        get_separator.argtypes = []
        self.sep = get_separator()

    def _raise_for_error(self, rc: int, err):
        if rc != 0:
            if err.contents:
                msg = self._get_error_msg(err.contents).decode("utf-8")
                exception_type = self._get_error_type(err.contents)
                self._free_error(err.contents)
                exn = _exception_types[exception_type]
                raise exn(msg)
            else:
                raise RuntimeError(f"Unknown error, function returned code {rc}")

    @contextlib.contextmanager
    def _get_model(self, data, seed):
        err = ctypes.pointer(ctypes.c_void_p())

        model = self._create_model(
            encode_stan_json(data),
            seed,
            print_callback if self.capture_stan_prints else None,
            err,
        )
        self._raise_for_error(not model, err)
        try:
            yield model
        finally:
            self._delete_model(model)

    def _encode_inits(self, inits, chains, seed):
        inits_encoded = None
        if inits is not None:
            if isinstance(inits, StanOutput):
                inits = inits.create_inits(chains=chains, seed=seed)

            if isinstance(inits, list):
                inits_encoded = self.sep.join(encode_stan_json(init) for init in inits)
            else:
                inits_encoded = encode_stan_json(inits)
        return inits_encoded

    def _get_parameter_names(self, model):
        comma_separated = self._get_param_names(model).decode("utf-8").strip()
        if comma_separated == "":
            return []
        return list(comma_separated.split(","))

    def api_version(self):
        """Return the version of the TinyStan API backing this model."""
        major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        return (major.value, minor.value, patch.value)

    def stan_version(self):
        """Return the version of Stan backing this model."""
        major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self._stan_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        return (major.value, minor.value, patch.value)

    def sample(
        self,
        data: StanData = "",
        *,
        num_chains: int = 4,
        inits: Union[StanData, List[StanData], None] = None,
        seed: Optional[int] = None,
        id: int = 1,
        init_radius: float = 2.0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        metric: HMCMetric = HMCMetric.DIAGONAL,
        init_inv_metric: Optional[np.ndarray] = None,
        save_inv_metric: bool = False,
        adapt: bool = True,
        delta: float = 0.8,
        gamma: float = 0.05,
        kappa: float = 0.75,
        t0: float = 10,
        init_buffer: int = 75,
        term_buffer: int = 50,
        window: int = 25,
        save_warmup: bool = False,
        stepsize: float = 1.0,
        stepsize_jitter: float = 0.0,
        max_depth: int = 10,
        refresh: int = 0,
        num_threads: int = -1,
    ):
        """
        Run Stan's No-U-Turn Sampler (NUTS) to sample from the posterior.
        An in-depth explanation of the parameters can be found in the Stan
        documentation at https://mc-stan.org/docs/reference-manual/mcmc.html

        Parameters
        ----------
        data : str | dict, optional
            The data to use for the model. This can be a
            path to a JSON file, a JSON string, or a dictionary.
            By default, ""
        num_chains : int, optional
            The number of chains to run, by default 4
        inits : str | dict | list[str | dict] | None, optional
            Initial parameter values. This can be a single
            path to a JSON file, a JSON string, a dictionary, or a
            list of length ``num_chains`` of those.
            By default, ""
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.
        id : int, optional
            Chain ID for the first chain, by default 1
        init_radius : float, optional
            Radius to initialize unspecified parameters within.
            The parameter values are drawn uniformly from the interval
            [-init_radius, init_radius] on the unconstrained scale.
            By default 2.0
        num_warmup : int, optional
            Number of warmup iterations to run, by default 1000
        num_samples : int, optional
            Number of samples to draw after warmup, by default 1000
        metric : HMCMetric, optional
            The type of inverse mass matrix to use in the sampler.
            The options are ``UNIT``, ``DENSE``, and ``DIAGONAL``.
            By default HMCMetric.DIAGONAL
        init_inv_metric : Optional[np.ndarray], optional
            Initial value for the inverse mass matrix used by the sampler.
            Valid shapes depend on the value of ``metric``. Can have
            a leading dimension of ``num_chains`` to specify different
            initial metrics for each chain.
        save_inv_metric : bool, optional
            Whether to report the final inverse mass matrix, by default False
        adapt : bool, optional
            Whether the sampler should adapt the step size and metric,
            by default True
        delta : float, optional
            Target average acceptance probability, by default 0.8
        gamma : float, optional
            Adaptation regularization scale, by default 0.05
        kappa : float, optional
            Adaptation relaxation exponent, by default 0.75
        t0 : float, optional
            Adaptation iteration offset, by default 10
        init_buffer : int, optional
            Number of warmup samples to use for initial step size adaptation,
            by default 75
        term_buffer : int, optional
            Number of warmup samples to use for step size adaptation
            after the metric is adapted, by default 50
        window : int, optional
            Initial number of iterations to use for metric adaptation,
            which is doubled each time the adaptation window is hit,
            by default 25
        save_warmup : bool, optional
            Whether to save the warmup samples, by default False
        stepsize : float, optional
            Initial step size for the sampler, by default 1.0
        stepsize_jitter : float, optional
            Amount of random jitter to add to the step size, by default 0.0
        max_depth : int, optional
            Maximum tree depth for the sampler, by default 10
        refresh : int, optional
            Number of iterations between progress messages, by default 0
            (supress messages)
        num_threads : int, optional
            Number of threads to use for sampling, by default -1
            (use all available)

        Returns
        -------
        StanOutput
            An object containing the samples and metadata from the sampling run.

        Raises
        ------
        ValueError
            If any of the parameters are invalid or out of range.
        RuntimeError
            If there is an unrecoverable error during sampling.
        """
        # these are checked here because they're sizes for "out"
        if num_chains < 1:
            raise ValueError("num_chains must be at least 1")
        if num_warmup < 0:
            raise ValueError("num_warmup must be non-negative")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")

        seed = seed or rand_u32()

        with self._get_model(data, seed) as model:
            model_params = self._num_free_params(model)

            param_names = HMC_SAMPLER_VARIABLES + self._get_parameter_names(model)

            num_params = len(param_names)
            num_draws = num_samples + num_warmup * save_warmup
            out = np.zeros((num_chains, num_draws, num_params), dtype=np.float64)

            metric_size = (
                (model_params, model_params)
                if metric == HMCMetric.DENSE
                else (model_params,)
            )

            if init_inv_metric is not None:
                if init_inv_metric.shape == metric_size:
                    init_inv_metric = np.repeat(
                        init_inv_metric[np.newaxis], num_chains, axis=0
                    )
                elif init_inv_metric.shape == (num_chains, *metric_size):
                    pass
                else:
                    raise ValueError(
                        f"Invalid initial metric size. Expected a {metric_size} "
                        f"or {(num_chains, *metric_size)} matrix."
                    )

            stepsize_out = None
            inv_metric_out = None

            if adapt:
                stepsize_out = np.zeros(num_chains, dtype=np.float64)
                if save_inv_metric:
                    inv_metric_out = np.zeros(
                        (num_chains, *metric_size), dtype=np.float64
                    )

            err = ctypes.pointer(ctypes.c_void_p())
            rc = self._ffi_sample(
                model,
                num_chains,
                self._encode_inits(inits, num_chains, seed),
                seed,
                id,
                init_radius,
                num_warmup,
                num_samples,
                metric.value,
                init_inv_metric,
                adapt,
                delta,
                gamma,
                kappa,
                t0,
                init_buffer,
                term_buffer,
                window,
                save_warmup,
                stepsize,
                stepsize_jitter,
                max_depth,
                refresh,
                num_threads,
                out,
                out.size,
                stepsize_out,
                inv_metric_out,
                err,
            )
            self._raise_for_error(rc, err)

        output = StanOutput(param_names, out)
        output.stepsize = stepsize_out
        output.inv_metric = inv_metric_out

        return output

    def pathfinder(
        self,
        data: StanData = "",
        *,
        num_paths: int = 4,
        inits: Optional[StanData] = None,
        seed: Optional[int] = None,
        id: int = 1,
        init_radius: float = 2.0,
        num_draws: int = 1000,
        max_history_size: int = 5,
        init_alpha: float = 0.001,
        tol_obj: float = 1e-12,
        tol_rel_obj: float = 1e4,
        tol_grad: float = 1e-8,
        tol_rel_grad: float = 1e7,
        tol_param: float = 1e-8,
        num_iterations: int = 1000,
        num_elbo_draws: int = 25,
        num_multi_draws: int = 1000,
        calculate_lp: bool = True,
        psis_resample: bool = True,
        refresh: int = 0,
        num_threads: int = -1,
    ):
        """
        Run the Pathfinder algorithm to approximate the posterior.
        See https://mc-stan.org/docs/reference-manual/pathfinder.html
        for more information on the algorithm.

        Parameters
        ----------
        data : str | dict, optional
            The data to use for the model. This can be a
            path to a JSON file, a JSON string, or a dictionary.
            By default, ""
        num_paths : int, optional
            The number of individual runs of the algorithm to run in parallel, by default 4
        inits : str | dict | list[str | dict] | None, optional
            Initial parameter values. This can be a single
            path to a JSON file, a JSON string, a dictionary, or a
            list of length ``num_paths`` of those.
            By default, ""
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.
        id : int, optional
            ID for the first path, by default 1
        init_radius : float, optional
            Radius to initialize unspecified parameters within.
            The parameter values are drawn uniformly from the interval
            [-init_radius, init_radius] on the unconstrained scale.
            By default 2.0
        num_draws : int, optional
            Number of approximate draws drawn from each of the
            ``num_paths`` Pathfinders, by default 1000
        max_history_size : int, optional
            History size used by the internal L-BFGS algorithm to
            approximate the Hessian, by default 5
        init_alpha : float, optional
            Initial step size for the internal L-BFGS algorithm,
            by default 0.001
        tol_obj : float, optional
            Convergence tolerance for the objective function for
            the internal L-BFGS algorithm, by default 1e-12
        tol_rel_obj : float, optional
            Relative convergence tolerance for the objective function
            for the internal L-BFGS algorithm, by default 1e4
        tol_grad : float, optional
            Convergence tolerance for the gradient norm for the internal
            L-BFGS algorithm, by default 1e-8
        tol_rel_grad : float, optional
            Relative convergence tolerance for the gradient norm for the
            internal L-BFGS algorithm, by default 1e7
        tol_param : float, optional
            Convergence tolerance for the changes in parameters for the
            internal L-BFGS algorithm, by default 1e-8
        num_iterations : int, optional
            Maximum number of iterations for the internal L-BFGS algorithm,
            by default 1000
        num_elbo_draws : int, optional
            Number of Monte Carlo draws used to estimate the ELBO,
            by default 25
        num_multi_draws : int, optional
            Number of draws returned by Multi-Pathfinder, by default 1000
        calculate_lp : bool, optional
            Whether to calculate the log probability of the approximate draws.
            If False, this also implies ``psis_resample=False``. By default True
        psis_resample : bool, optional
            Whether to use Pareto smoothed importance sampling on
            the approximate draws. If False, all ``num_path * num_draws``
            approximate samples will be returned. By default True.
        refresh : int, optional
            Number of iterations between progress messages, by default 0
            (supress messages)
        num_threads : int, optional
            Number of threads to use for Pathfinder, by default -1
            (use all available)

        Returns
        -------
        StanOutput
            An object containing the samples and metadata from the algorithm.

        Raises
        ------
        ValueError
            If any of the parameters are invalid or out of range.
        RuntimeError
            If there is an unrecoverable error during the algorithm.
        """
        if num_draws < 1:
            raise ValueError("num_draws must be at least 1")
        if num_paths < 1:
            raise ValueError("num_paths must be at least 1")
        if num_multi_draws < 1:
            raise ValueError("num_multi_draws must be at least 1")

        if calculate_lp and psis_resample:
            output_size = num_multi_draws
        else:
            output_size = num_draws * num_paths

        seed = seed or rand_u32()

        with self._get_model(data, seed) as model:
            model_params = self._num_free_params(model)
            if model_params == 0:
                raise ValueError("Model has no parameters.")

            param_names = PATHFINDER_VARIABLES + self._get_parameter_names(model)

            num_params = len(param_names)
            out = np.zeros((output_size, num_params), dtype=np.float64)

            err = ctypes.pointer(ctypes.c_void_p())
            rc = self._ffi_pathfinder(
                model,
                num_paths,
                self._encode_inits(inits, num_paths, seed),
                seed,
                id,
                init_radius,
                num_draws,
                max_history_size,
                init_alpha,
                tol_obj,
                tol_rel_obj,
                tol_grad,
                tol_rel_grad,
                tol_param,
                num_iterations,
                num_elbo_draws,
                num_multi_draws,
                calculate_lp,
                psis_resample,
                refresh,
                num_threads,
                out,
                out.size,
                err,
            )
            self._raise_for_error(rc, err)

        return StanOutput(param_names, out)

    def optimize(
        self,
        data: StanData = "",
        *,
        init: Optional[StanData] = None,
        seed: Optional[int] = None,
        id: int = 1,
        init_radius: float = 2.0,
        algorithm: OptimizationAlgorithm = OptimizationAlgorithm.LBFGS,
        jacobian: bool = False,
        num_iterations: int = 2000,
        max_history_size: int = 5,
        init_alpha: float = 0.001,
        tol_obj: float = 1e-12,
        tol_rel_obj: float = 1e4,
        tol_grad: float = 1e-8,
        tol_rel_grad: float = 1e7,
        tol_param: float = 1e-8,
        refresh: int = 0,
        num_threads: int = -1,
    ):
        """
        Optimize the model parameters using the specified algorithm.

        This will find either the maximum a posteriori (MAP) estimate
        or the maximum likelihood estimate (MLE) of the model parameters,
        depending on the value of the ``jacobian`` parameter.
        Additional parameters can be found in the Stan documentation at
        https://mc-stan.org/docs/reference-manual/optimization.html

        Parameters
        ----------
        data : str | dict, optional
            The data to use for the model. This can be a
            path to a JSON file, a JSON string, or a dictionary.
            By default, ""
        init : str | dict | None, optional
            Initial parameter values. This can be a
            path to a JSON file, a JSON string, or a dictionary.
            By default, ""
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.
        id : int, optional
            ID used to offset the random number generator, by default 1
        init_radius : float, optional
            Radius to initialize unspecified parameters within.
            The parameter values are drawn uniformly from the interval
            [-init_radius, init_radius] on the unconstrained scale.
            By default 2.0
        algorithm : OptimizationAlgorithm, optional
            Which optimization algorithm to use. Some of the following
            arguments may be ignored depending on the algorithm.
            By default OptimizationAlgorithm.LBFGS
        jacobian : bool, optional
            Whether to apply the Jacobian change of variables to the
            log density. If False, the algorithm will find the MLE.
            If True, the algorithm will find the MAP estimate.
            By default False
        num_iterations : int, optional
            Maximum number of iterations to run the optimization,
            by default 2000
        max_history_size : int, optional
            History size used to approximate the Hessian, by default 5
        init_alpha : float, optional
            Initial step size, by default 0.001
        tol_obj : float, optional
            Convergence tolerance for the objective function,
            by default 1e-12
        tol_rel_obj : float, optional
            Relative convergence tolerance for the objective function,
            by default 1e4
        tol_grad : float, optional
            Convergence tolerance for the gradient norm, by default 1e-8
        tol_rel_grad : float, optional
            Relative convergence tolerance for the gradient norm,
            by default 1e7
        tol_param : float, optional
            Convergence tolerance for the changes in parameters,
            by default 1e-8
        refresh : int, optional
            Number of iterations between progress messages, by default 0
            (supress messages)
        num_threads : int, optional
            Number of threads to use for log density evaluations, by default -1
            (use all available)

        Returns
        -------
        StanOutput
            An object containing the samples and metadata from the algorithm.

        Raises
        ------
        ValueError
            If any of the parameters are invalid or out of range.
        RuntimeError
            If there is an unrecoverable error during the algorithm.
        """
        seed = seed or rand_u32()

        with self._get_model(data, seed) as model:
            param_names = OPTIMIZE_VARIABLES + self._get_parameter_names(model)

            num_params = len(param_names)
            out = np.zeros(num_params, dtype=np.float64)

            err = ctypes.pointer(ctypes.c_void_p())
            rc = self._ffi_optimize(
                model,
                self._encode_inits(init, 1, seed),
                seed,
                id,
                init_radius,
                algorithm.value,
                num_iterations,
                jacobian,
                max_history_size,
                init_alpha,
                tol_obj,
                tol_rel_obj,
                tol_grad,
                tol_rel_grad,
                tol_param,
                refresh,
                num_threads,
                out,
                out.size,
                err,
            )
            self._raise_for_error(rc, err)

        return StanOutput(param_names, out)

    def laplace_sample(
        self,
        mode: Union[StanOutput, np.ndarray, StanData],
        data: StanData = "",
        *,
        seed: Optional[int] = None,
        num_draws: int = 1000,
        jacobian: bool = True,
        calculate_lp: bool = True,
        save_hessian: bool = False,
        refresh: int = 0,
        num_threads: int = -1,
    ):
        """
        Sample from the Laplace approximation of the posterior
        centered at the provided mode.

        Parameters
        ----------
        mode : Union[StanOutput, np.ndarray, StanData]
            The mode of the Laplace approximation. This can be a
            StanOutput object from :meth:`~Model.optimize`, a numpy
            array, a path to a JSON file, a JSON string, or a dictionary.
        data : str | dict, optional
            The data to use for the model. This can be a
            path to a JSON file, a JSON string, or a dictionary.
            By default, ""
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.
        num_draws : int, optional
            Number of draws, by default 1000
        jacobian : bool, optional
            Whether to apply the Jacobian change of variables to the
            log density. **Note:** This should match the value used
            when the mode was calculated.
            By default True.
        calculate_lp : bool, optional
            Whether to calculate the log probability of the samples,
            by default True
        save_hessian : bool, optional
            Whether to save the Hessian matrix calculated at the mode,
            by default False
        refresh : int, optional
            Number of iterations between progress messages, by default 0
            (supress messages)
        num_threads : int, optional
            Number of threads to use for log density evaluations, by default -1
            (use all available)

        Returns
        -------
        StanOutput
            An object containing the samples and metadata from the algorithm.

        Raises
        ------
        ValueError
            If any of the parameters are invalid or out of range.
        RuntimeError
            If there is an unrecoverable error during the algorithm.
        """
        if num_draws < 1:
            raise ValueError("num_draws must be at least 1")

        seed = seed or rand_u32()

        mode_array, mode_json = preprocess_laplace_inputs(mode)

        with self._get_model(data, seed) as model:
            req_params = self._num_req_constrained_params(model)
            if mode_array is not None and len(mode_array) < req_params:
                raise ValueError(
                    "Mode array has incorrect length. "
                    f"Expected at least {req_params} but got {len(mode_array)}"
                )

            param_names = LAPLACE_VARIABLES + self._get_parameter_names(model)
            num_params = len(param_names)
            out = np.zeros((num_draws, num_params), dtype=np.float64)

            model_params = self._num_free_params(model)
            hessian_out = (
                np.zeros((model_params, model_params), dtype=np.float64)
                if save_hessian
                else None
            )
            err = ctypes.pointer(ctypes.c_void_p())

            rc = self._ffi_laplace(
                model,
                mode_array,
                mode_json,
                seed,
                num_draws,
                jacobian,
                calculate_lp,
                refresh,
                num_threads,
                out,
                out.size,
                hessian_out,
                err,
            )
            self._raise_for_error(rc, err)

        output = StanOutput(param_names, out)
        if save_hessian:
            output.hessian = hessian_out
        return output
