# FFIStan

**FFIStan** (short for *Foreign Function Interface Stan*, alternatively *Fast, Flatiron Institute, Stan*) is a library that
provides a C interface to the [Stan](https://mc-stan.org) algorithms, such as sampling with the No-U-Turn Sampler (NUTS).

It is intended to be a counterpart to [BridgeStan](https://github.com/roualdes/bridgestan), which provides a C interface to the methods of a Stan *model*.

Similar packages are [PyStan](https://mc-stan.org/users/interfaces/pystan) and [RStan](https://mc-stan.org/users/interfaces/rstan).
FFIStan differs from those packages by working at a lower level of interaction between the user's language and Stan, in a way
which should be more portable. No language-specific code is required to compile FFIStan, and the same model can be used from
any of the supported languages.
