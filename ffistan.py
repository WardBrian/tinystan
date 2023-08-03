import ctypes
import subprocess
from enum import Enum

import numpy as np
from numpy.ctypeslib import ndpointer

double_array = ndpointer(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
err_ptr = ctypes.POINTER(ctypes.c_void_p)

HMC_SAMPLER_VARIABLES = [
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
]

FIXED_SAMPLER_VARIABLES = [
    "lp__",
    "accept_stat__",
]


class Metric(Enum):
    """Docstring for Metric."""

    UNIT = 0
    DENSE = 1
    DIAG = 2


class FFIStanModel:
    def __init__(self, model):
        if model.endswith(".stan"):
            libname = model[:-5] + "_model.so"
            subprocess.run(["make", "STAN_THREADS=true", libname])
            self._lib = ctypes.CDLL(libname)
        else:
            self._lib = ctypes.CDLL(model)

        self._create_model = self._lib.ffistan_create_model
        self._create_model.restype = ctypes.c_void_p
        self._create_model.argtypes = [ctypes.c_char_p, ctypes.c_uint, err_ptr]

        self._delete_model = self._lib.ffistan_destroy_model
        self._delete_model.restype = None
        self._delete_model.argtypes = [ctypes.c_void_p]

        self._get_names = self._lib.ffistan_model_param_names
        self._get_names.restype = ctypes.c_char_p
        self._get_names.argtypes = [ctypes.c_void_p]

        self._ffi_sample = self._lib.ffistan_sample
        self._ffi_sample.restype = ctypes.c_int
        self._ffi_sample.argtypes = [
            ctypes.c_void_p,  # model
            ctypes.c_size_t,  # num_chains
            ctypes.c_char_p,  # inits
            ctypes.c_uint,  # seed
            ctypes.c_uint,  # chain_id
            ctypes.c_double,  # init_radius
            ctypes.c_int,  # num_warmup
            ctypes.c_int,  # num_samples
            ctypes.c_int,  # really enum for metric
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
            ctypes.c_int,  # refresh
            ctypes.c_double,  # stepsize
            ctypes.c_double,  # stepsize_jitter
            ctypes.c_int,  # max_depth
            double_array,
            err_ptr,
        ]

        self._get_error = self._lib.ffistan_get_error_message
        self._get_error.restype = ctypes.c_char_p
        self._get_error.argtypes = [ctypes.c_void_p]
        self._free_error = self._lib.ffistan_free_stan_error
        self._free_error.restype = None
        self._free_error.argtypes = [ctypes.c_void_p]
        get_separator = self._lib.ffistan_separator_char
        get_separator.restype = ctypes.c_char
        get_separator.argtypes = []
        self.sep = get_separator().decode("utf-8")

    def _raise_for_error(self, rc: int, err: ctypes.pointer):
        if rc != 0:
            if err.contents:
                msg = self._get_error(err.contents).decode("utf-8")
                self._free_error(err.contents)
                raise RuntimeError(msg)
            else:
                raise RuntimeError(f"Unknown error, function returned code {rc}")

    def sample(
        self,
        data,
        *,
        num_chains=4,
        inits=None,
        seed=None,
        chain_id=1,
        init_radius=2.0,
        num_warmup=1000,
        num_samples=1000,
        metric=Metric.DIAG,
        adapt=True,
        delta=0.8,
        gamma=0.05,
        kappa=0.75,
        t0=10,
        init_buffer=75,
        term_buffer=50,
        window=25,
        save_warmup=False,
        refresh=0,
        stepsize=1.0,
        stepsize_jitter=0.0,
        max_depth=10,
    ):
        assert num_chains > 0, "num_chains must be positive"
        assert num_warmup >= 0, "num_warmup must be non-negative"
        assert num_samples > 0, "num_samples must be positive"

        seed = seed or np.random.randint(2**32 - 1)
        err = ctypes.pointer(ctypes.c_void_p())

        model = self._create_model(data.encode(), seed, err)
        self._raise_for_error(not model, err)

        param_names = HMC_SAMPLER_VARIABLES + list(
            self._get_names(model).decode("utf-8").split(",")
        )

        num_params = len(param_names)
        num_draws = num_samples + num_warmup * save_warmup
        out = np.zeros((num_chains, num_draws, num_params), dtype=np.float64)

        inits_encoded = None
        if inits is not None:
            if isinstance(inits, list):
                inits_encoded = self.sep.join(inits).encode()
            else:
                inits_encoded = inits.encode()

        rc = self._ffi_sample(
            model,
            num_chains,
            inits_encoded,
            seed,
            chain_id,
            init_radius,
            num_warmup,
            num_samples,
            metric.value,
            adapt,
            delta,
            gamma,
            kappa,
            t0,
            init_buffer,
            term_buffer,
            window,
            save_warmup,
            refresh,
            stepsize,
            stepsize_jitter,
            max_depth,
            out,
            err,
        )

        self._delete_model(model)
        self._raise_for_error(rc, err)

        return (param_names, out)


if __name__ == "__main__":
    import os

    os.environ["STAN_NUM_THREADS"] = "-1"

    model = FFIStanModel("./bernoulli.stan")
    data = "bernoulli.data.json"
    fit = model.sample(data, num_samples=10000, num_chains=10)

    print(fit[0])
    print(fit[1].mean(axis=(0, 1))[7])
    print(fit[1].shape)
