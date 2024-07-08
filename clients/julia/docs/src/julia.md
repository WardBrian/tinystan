# Julia Interface

```@raw html
% NB: If you are reading this file in docs/languages, you are reading a generated output!
% This should be apparent due to the html tags everywhere.
% If you are reading this in julia/docs/src, you are reading the true source!
% Please only make edits in the later file, since the first is DELETED each re-build.
```

---

## Installation

### From Source
This section assumes you have followed the [Getting Started guide](../getting-started.rst)
to install TinyStan's pre-requisites and downloaded a copy of the TinyStan source code.

To install the Julia interface, you can either install it directly from Github by running
the following inside a Julia REPL

```julia
] add https://github.com/WardBrian/tinystan.git:clients/julia
```

Or, since you have already downloaded the repository, you can run

```julia
] dev clients/julia/
```

from the TinyStan folder.

The first time you compile a model, the TinyStan source code for your current version
will be downloaded to a hidden directory in the users `HOME` directory.

To use the TinyStan source you've manually downloaded instead of
one the package will download for you, you must use
[`set_tinystan_path()`](TinyStan.set_tinystan_path!) or the `$TINYSTAN`
environment variable.

Note that the Julia package depends on Julia 1.6+ and the `Inflate` package.



## Example Program

An example program is provided alongside the Julia interface code in `example.jl`:


```@raw html
<details>
<summary><a>Show example.jl</a></summary>
```

```{literalinclude} ../../clients/julia/example.jl
:language: julia
```

```@raw html
</details>
```


## API Reference

### Model interface

```@docs
Model
sample
HMCMetric
pathfinder
optimize
OptimizationAlgorithm
laplace_sample
```

### Compilation utilities
```@docs
compile_model
get_tinystan_path
set_tinystan_path!
```
