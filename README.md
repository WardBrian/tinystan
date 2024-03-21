# TinyStan

**Note**: This project is still under active development. The API is not yet stable, and the documentation is incomplete. Using before a 1.0 release is not recommended for most users.

**TinyStan** is a library that provides a C interface to the [Stan](https://mc-stan.org) algorithms, such as sampling with the No-U-Turn Sampler (NUTS),
and language bindings in Julia, Python, and R.

It is intended to be a counterpart to [BridgeStan](https://github.com/roualdes/bridgestan), which provides a C interface to the methods of a Stan *model*.

Similar packages are [PyStan](https://mc-stan.org/users/interfaces/pystan) and [RStan](https://mc-stan.org/users/interfaces/rstan).
TinyStan differs from those packages by working at a lower level of interaction between the user's language and Stan, in a way
which should be more portable. No language-specific code is required to compile TinyStan, and the same model can be used from
any of the supported languages.
