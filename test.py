import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

double_array = ndpointer(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
err_ptr = ctypes.POINTER(ctypes.c_void_p)
c_print_callback = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.c_int)


m = ctypes.CDLL("./bernoulli_model.so")


sample = m.cstan_sample
sample.restype = ctypes.c_int
sample.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    double_array,
    err_ptr,
]

get_error = m.cstan_get_error_message
get_error.restype = ctypes.c_char_p
get_error.argtypes = [ctypes.c_void_p]
free_error = m.cstan_free_stan_error
free_error.restype = None
free_error.argtypes = [ctypes.c_void_p]

def raise_for_error(err: ctypes.pointer):
    if err.contents:
        msg = get_error(err.contents).decode("utf-8")
        free_error(err.contents)
        raise RuntimeError(msg)

err = ctypes.pointer(ctypes.c_void_p())

# 8 * num_samples + 8 * num_warmup * save_warmup
x = np.zeros((100,8), dtype=np.float64)
data = 'bernoulli.data.json'.encode()

print((x == 0).all())
sample(data, None, 1234, 1, 2.0, 500, 100, False, 10, 0.1, 0.1, 8, x, err)
raise_for_error(err)
print(x[:,7].mean())
