import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

double_array = ndpointer(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
err_ptr = ctypes.POINTER(ctypes.c_void_p)
c_print_callback = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.c_int)


m = ctypes.CDLL("./bernoulli_model.so")


# TODO: intercept handling (Ctrl+C)?

ffi_sample = m.ffistan_sample
ffi_sample.restype = ctypes.c_int
ffi_sample.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,  # really enum for metric
    # adaptation
    ctypes.c_bool,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    double_array,
    err_ptr,
]

get_error = m.ffistan_get_error_message
get_error.restype = ctypes.c_char_p
get_error.argtypes = [ctypes.c_void_p]
free_error = m.ffistan_free_stan_error
free_error.restype = None
free_error.argtypes = [ctypes.c_void_p]


def raise_for_error(err: ctypes.pointer):
    if err.contents:
        msg = get_error(err.contents).decode("utf-8")
        free_error(err.contents)
        raise RuntimeError(msg)

from enum import Enum


class Metric(Enum):
    """Docstring for Metric."""
    UNIT = 0
    DENSE = 1
    DIAG = 2


def sample(
    data,
    *,
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

    assert num_warmup >= 0, "num_warmup must be non-negative"
    assert num_samples > 0, "num_samples must be positive"

    seed = seed or np.random.randint(2**32 - 1)

    constrained_params = 1  # TODO
    num_params = 7 + constrained_params
    num_draws = num_samples + num_warmup * save_warmup
    out = np.zeros((num_draws, num_params), dtype=np.float64)

    err = ctypes.pointer(ctypes.c_void_p())

    ffi_sample(
        data.encode(),
        inits.encode() if inits else None,
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

    raise_for_error(err)
    return out


# need some way of determining the number 8 - could use param_unc_num from BS, but requires data?

# 8 = 7 (sampler parameters) + 1 constrained parameter from bernoulli model
# 8 * num_samples + 8 * num_warmup * save_warmup
data = "bernoulli.data.json"

fit = sample(data, num_samples=100, metric=Metric.UNIT, save_warmup=True)

print(fit[1000:, 7].mean())
print(fit.shape)
