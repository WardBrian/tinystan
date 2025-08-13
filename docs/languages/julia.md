
<a id='Julia-Interface'></a>

<a id='Julia-Interface-1'></a>

# Julia Interface


% NB: If you are reading this file in docs/languages, you are reading a generated output!
% This should be apparent due to the html tags everywhere.
% If you are reading this in julia/docs/src, you are reading the true source!
% Please only make edits in the later file, since the first is DELETED each re-build.


---


<a id='Installation'></a>

<a id='Installation-1'></a>

## Installation


<a id='From-Source'></a>

<a id='From-Source-1'></a>

### From Source


This section assumes you have followed the [Getting Started guide](../getting-started.rst) to install TinyStan's pre-requisites and downloaded a copy of the TinyStan source code.


To install the Julia interface, you can either install it directly from Github by running the following inside a Julia REPL


```julia
] add https://github.com/WardBrian/tinystan.git:clients/julia
```


Or, since you have already downloaded the repository, you can run


```julia
] dev clients/julia/
```


from the TinyStan folder.


The first time you compile a model, the TinyStan source code for your current version will be downloaded to a hidden directory in the users `HOME` directory.


To use the TinyStan source you've manually downloaded instead of one the package will download for you, you must use [`set_tinystan_path()`](TinyStan.set_tinystan_path!) or the `$TINYSTAN` environment variable.


Note that the Julia package depends on Julia 1.6+ and the `Inflate` package.


<a id='Example-Program'></a>

<a id='Example-Program-1'></a>

## Example Program


An example program is provided alongside the Julia interface code in `example.jl`:


<details>
<summary><a>Show example.jl</a></summary>


```{literalinclude} ../../clients/julia/example.jl
:language: julia
```


</details>


<a id='API-Reference'></a>

<a id='API-Reference-1'></a>

## API Reference


<a id='Model-interface'></a>

<a id='Model-interface-1'></a>

### Model interface

<a id='TinyStan.Model' href='#TinyStan.Model'>#</a>
**`TinyStan.Model`** &mdash; *Type*.



```julia
Model(model::String; stanc_args::Vector{String} = String[], make_args::Vector{String} = String[], warn::Bool = true)
```

Load a Stan model for inference, compiling it if necessary.

If model is a path to a file ending in `.stan`, this will first compile the model.  Compilation occurs if no shared object file exists for the supplied Stan file or if a shared object file exists and the Stan file has changed since last compilation.  This is equivalent to calling [`compile_model`](julia.md#TinyStan.compile_model) and then the constructor. If `warn` is false, the warning about re-loading the same shared objects is suppressed.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L43-L54' class='documenter-source'>source</a><br>

<a id='TinyStan.StanOutput' href='#TinyStan.StanOutput'>#</a>
**`TinyStan.StanOutput`** &mdash; *Type*.



```julia
StanOutput
```

A structure to hold the output from a Stan model run. Always contains the names of the parameters and a "draws" array with algorithm's output. Depending on the algorithm, it may also contain `stepsize`, `inv_metric`, and `hessian` fields.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/output.jl#L2-L9' class='documenter-source'>source</a><br>

<a id='TinyStan.get_draws' href='#TinyStan.get_draws'>#</a>
**`TinyStan.get_draws`** &mdash; *Function*.



```julia
get_draws(output::StanOutput, name::String)
```

Returns the draws for a specific parameter from the `StanOutput` object.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/output.jl#L18-L22' class='documenter-source'>source</a><br>

<a id='TinyStan.sample' href='#TinyStan.sample'>#</a>
**`TinyStan.sample`** &mdash; *Function*.



```julia
sample(model::Model, data::String=""; num_chains::Int=4, inits::Union{nothing,AbstractString,AbstractArray{AbstractString}}=nothing, seed::Union{Nothing,UInt32}=nothing, id::Int=1, init_radius=2.0, num_warmup::Int=1000, num_samples::Int=1000, metric::HMCMetric=DIAGONAL, init_inv_metric::Union{Nothing,Array{Float64}}=nothing, save_inv_metric::Bool=false, adapt::Bool=true, delta::Float64=0.8, gamma::Float64=0.05, kappa::Float64=0.75, t0::Int=10, init_buffer::Int=75, term_buffer::Int=50, window::Int=25, save_warmup::Bool=false, stepsize::Float64=1.0, stepsize_jitter::Float64=0.0, max_depth::Int=10, refresh::Int=0, num_threads::Int=-1)
```

Run Stan's No-U-Turn Sampler (NUTS) to sample from the posterior. An in-depth explanation of the parameters can be found in the [Stan documentation](https://mc-stan.org/docs/reference-manual/mcmc.html).

Returns StanOutput object with the draws, parameter names, and adapted stepsizes. If `save_inv_metric` is true, the inverse metric is also returned.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L198-L208' class='documenter-source'>source</a><br>

<a id='TinyStan.HMCMetric' href='#TinyStan.HMCMetric'>#</a>
**`TinyStan.HMCMetric`** &mdash; *Type*.



Choices for the structure of the mass matrix used in the HMC sampler.

Either `UNIT`, `DENSE`, or `DIAGONAL`.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L3-L7' class='documenter-source'>source</a><br>

<a id='TinyStan.pathfinder' href='#TinyStan.pathfinder'>#</a>
**`TinyStan.pathfinder`** &mdash; *Function*.



```julia
pathfinder(model::Model, data::String=""; num_paths::Int=4, inits::Union{nothing,AbstractString,AbstractArray{AbstractString}}=nothing, seed::Union{Nothing,UInt32}=nothing, id::Int=1, init_radius=2.0, num_draws::Int=1000, max_history_size::Int=5, init_alpha::Float64=0.001, tol_obj::Float64=1e-12, tol_rel_obj::Float64=1e4, tol_grad::Float64=1e-8, tol_rel_grad::Float64=1e7, tol_param::Float64=1e-8, num_iterations::Int=1000, num_elbo_draws::Int=25, num_multi_draws::Int=1000, calculate_lp::Bool=true, psis_resample::Bool=true, refresh::Int=0, num_threads::Int=-1)
```

Run the Pathfinder algorithm to approximate the posterior. See [Stan's documentation](https://mc-stan.org/docs/reference-manual/pathfinder.html) for more information on the algorithm.

Returns StanOutput object with the draws, parameter names


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L346-L354' class='documenter-source'>source</a><br>

<a id='TinyStan.optimize' href='#TinyStan.optimize'>#</a>
**`TinyStan.optimize`** &mdash; *Function*.



```julia
optimize(model::Model, data::String=""; init::Union{Nothing,AbstractString}=nothing, seed::Union{UInt32,Nothing}=nothing, id::Int=1, init_radius::Float64=2.0, algorithm::OptimizationAlgorithm=LBFGS, jacobian::Bool=false, num_iterations::Int=2000, max_history_size::Int=5, init_alpha::Float64=0.001, tol_obj::Float64=1e-12, tol_rel_obj::Float64=1e4, tol_grad::Float64=1e-8, tol_rel_grad::Float64=1e7, tol_param::Float64=1e-8, refresh::Int=0, num_threads::Int=-1)
```

Optimize the model parameters using the specified algorithm.

This will find either the maximum a posteriori (MAP) estimate or the maximum likelihood estimate (MLE) of the model parameters, depending on the value of the `jacobian` parameter. Additional parameters can be found in the [Stan documentation](https://mc-stan.org/docs/reference-manual/optimization.html).

Returns StanOutput object with the draws, parameter names


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L443-L454' class='documenter-source'>source</a><br>

<a id='TinyStan.OptimizationAlgorithm' href='#TinyStan.OptimizationAlgorithm'>#</a>
**`TinyStan.OptimizationAlgorithm`** &mdash; *Type*.



Choices for the optimization algorithm to use.

Either `NEWTON`, `BFGS`, or `LBFGS`.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L14-L18' class='documenter-source'>source</a><br>

<a id='TinyStan.laplace_sample' href='#TinyStan.laplace_sample'>#</a>
**`TinyStan.laplace_sample`** &mdash; *Function*.



```julia
laplace_sample(model::Model, mode::Union{AbstractString,Array{Float64}}, data::AbstractString=""; num_draws::Int=1000, jacobian::Bool=true, calculate_lp::Bool=true, save_hessian::Bool=false, seed::Union{UInt32,Nothing}=nothing, refresh::Int=0, num_threads::Int=-1)
```

Sample from the Laplace approximation of the posterior centered at the provided mode. The mode can be either a JSON string or an array of floats, often obtained from the [`optimize`](julia.md#TinyStan.optimize) function.

Returns StanOutput object with the draws, parameter names. If `save_hessian` is true, the Hessian matrix is also returned.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/model.jl#L517-L527' class='documenter-source'>source</a><br>


<a id='Compilation-utilities'></a>

<a id='Compilation-utilities-1'></a>

### Compilation utilities

<a id='TinyStan.compile_model' href='#TinyStan.compile_model'>#</a>
**`TinyStan.compile_model`** &mdash; *Function*.



```julia
compile_model(stan_file; stanc_args=[], make_args=[])
```

Run TinyStan’s Makefile on a `.stan` file, creating the `.so` used by StanModel and return a path to the compiled library. Arguments to `stanc3` can be passed as a vector, for example `["--O1"]` enables level 1 compiler optimizations. Additional arguments to `make` can be passed as a vector, for example `["STAN_NO_RANGE_CHECKS=true"]` will disable bounds checking in the Stan Math library. If the same flags are defined in `make/local`, the versions passed here will take precedent.

This function checks that the path to TinyStan is valid and will error if it is not. This can be set with [`set_tinystan_path!()`](julia.md#TinyStan.set_tinystan_path!).


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/compile.jl#L61-L74' class='documenter-source'>source</a><br>

<a id='TinyStan.get_tinystan_path' href='#TinyStan.get_tinystan_path'>#</a>
**`TinyStan.get_tinystan_path`** &mdash; *Function*.



```julia
get_tinystan_path() -> String
```

Return the path the the TinyStan directory.

If the environment variable `TINYSTAN` is set, this will be returned. Otherwise, this function downloads a matching version of TinyStan under a folder called `.tinystan` in the user's home directory.

See [`set_tinystan_path!()`](julia.md#TinyStan.set_tinystan_path!) to set the path from within Julia.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/compile.jl#L27-L37' class='documenter-source'>source</a><br>

<a id='TinyStan.set_tinystan_path!' href='#TinyStan.set_tinystan_path!'>#</a>
**`TinyStan.set_tinystan_path!`** &mdash; *Function*.



```julia
set_tinystan_path!(path)
```

Set the path TinyStan.


<a target='_blank' href='https://github.com/WardBrian/TinyStan/blob/main/clients/julia/src/compile.jl#L17-L21' class='documenter-source'>source</a><br>

