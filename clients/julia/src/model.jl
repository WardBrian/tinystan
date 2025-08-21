using Base.Libc.Libdl: dlsym, dlopen, dllist

"""
Choices for the structure of the inverse mass matrix used in the HMC sampler.

Either `UNIT`, `DENSE`, or `DIAGONAL`.
"""
@enum HMCMetric begin
    UNIT = 0
    DENSE = 1
    DIAGONAL = 2
end

"""
Choices for the optimization algorithm to use.

Either `NEWTON`, `BFGS`, or `LBFGS`.
"""
@enum OptimizationAlgorithm begin
    NEWTON = 0
    BFGS = 1
    LBFGS = 2
end

const HMC_SAMPLER_VARIABLES = [
    "lp__",
    "accept_stat__",
    "stepsize__",
    "treedepth__",
    "n_leapfrog__",
    "divergent__",
    "energy__",
]

const PATHFINDER_VARIABLES = ["lp_approx__", "lp__", "path__"]

const OPTIMIZE_VARIABLES = ["lp__"]

const LAPLACE_VARIABLES = ["log_p__", "log_q__"]

const exceptions = [ErrorException, ArgumentError, _ -> InterruptException()]

"""
    Model(model::String; stanc_args::Vector{String} = String[], make_args::Vector{String} = String[], warn::Bool = true)

Load a Stan model for inference, compiling it if necessary.

If model is a path to a file ending in `.stan`, this will first compile
the model.  Compilation occurs if no shared object file exists for the
supplied Stan file or if a shared object file exists and the Stan file
has changed since last compilation.  This is equivalent to calling
[`compile_model`](@ref) and then the constructor. If `warn` is
false, the warning about re-loading the same shared objects is suppressed.
"""
mutable struct Model
    lib::Ptr{Nothing}
    const sep::Char

    function Model(
        model::String;
        stanc_args::AbstractVector{String} = String[],
        make_args::AbstractVector{String} = String[],
        warn::Bool = true,
    )

        if !isfile(model)
            throw(SystemError("File not found: $model"))
        end

        if endswith(model, ".stan")
            libname = compile_model(model; stanc_args, make_args)
        else
            libname = model
        end

        if warn && in(abspath(libname), dllist())
            @warn "Loading a shared object '" *
                  libname *
                  "' which is already loaded.\n" *
                  "If the file has changed since the last time it was loaded, this load may not update the library!"
        end

        windows_dll_path_setup()
        lib = dlopen(libname)

        major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
        @ccall $(dlsym(lib, :tinystan_api_version))(
            major::Ref{Cint},
            minor::Ref{Cint},
            patch::Ref{Cint},
        )::Cvoid
        api_ver = VersionNumber(major[], minor[], patch[])
        if api_ver.major != TinyStan.pkg_version.major
            error(
                "Incompatible TinyStan API version. " *
                "Expected $(TinyStan.pkg_version) but got $api_ver.\n" *
                "You need to re-compile your model.",
            )
        elseif api_ver != TinyStan.pkg_version
            @warn "TinyStan API version does not match. " *
                  "Expected $(TinyStan.pkg_version) but got $api_ver.\n" *
                  "You may need to re-compile your model."
        end

        sep = Char(@ccall $(dlsym(lib, :tinystan_separator_char))()::Cchar)
        new(lib, sep)
    end

end


# helper functions
function raise_for_error(lib::Ptr{Nothing}, return_code::Cint, err::Ref{Ptr{Cvoid}})
    if return_code != 0
        if err[] == C_NULL
            error("Unknown error, function returned code $return_code")
        end

        cstr = @ccall $(dlsym(lib, :tinystan_get_error_message))(err[]::Ptr{Cvoid})::Cstring

        msg = unsafe_string(cstr)
        type = @ccall $(dlsym(lib, :tinystan_get_error_type))(err[]::Ptr{Cvoid})::Cint
        @ccall $(dlsym(lib, :tinystan_destroy_error))(err[]::Ptr{Cvoid})::Cvoid
        exn = exceptions[type+1]
        throw(exn(msg))
    end
end

function encode_inits(sep::Char, inits::Union{String,Vector{String},Nothing})
    if inits === nothing
        return C_NULL
    end
    if inits isa String
        return inits
    end
    if inits isa Vector{String}
        return join(inits, sep)
    end
end

function with_model(f, model::Model, data::String, seed::UInt32)
    err = Ref{Ptr{Cvoid}}()
    model_ptr = @ccall $(dlsym(model.lib, :tinystan_create_model))(
        data::Cstring,
        seed::Cuint,
        C_NULL::Ptr{Cvoid},
        err::Ref{Ptr{Cvoid}},
    )::Ptr{Cvoid}

    raise_for_error(model.lib, Int32(model_ptr == C_NULL), err)

    try
        return f(model_ptr)
    finally
        @ccall $(dlsym(model.lib, :tinystan_destroy_model))(model_ptr::Ptr{Cvoid})::Cvoid
    end
end

function num_free_params(model::Model, model_ptr::Ptr{Cvoid})
    Int(
        @ccall $(dlsym(model.lib, :tinystan_model_num_free_params))(
            model_ptr::Ptr{Cvoid},
        )::Csize_t
    )
end

function get_names(model::Model, model_ptr::Ptr{Cvoid})
    cstr = @ccall $(dlsym(model.lib, :tinystan_model_param_names))(
        model_ptr::Ptr{Cvoid},
    )::Cstring
    str = unsafe_string(cstr)
    if isempty(str)
        return String[]
    end
    string.(split(unsafe_string(cstr), ','))
end

function api_version(model::Model)
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    @ccall $(dlsym(model.lib, :tinystan_api_version))(
        major::Ref{Cint},
        minor::Ref{Cint},
        patch::Ref{Cint},
    )::Cvoid
    (major[], minor[], patch[])
end

function stan_version(model::Model)
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    @ccall $(dlsym(model.lib, :tinystan_stan_version))(
        major::Ref{Cint},
        minor::Ref{Cint},
        patch::Ref{Cint},
    )::Cvoid
    (major[], minor[], patch[])
end

"""
    sample(model::Model, data::String=""; num_chains::Int=4, inits::Union{nothing,AbstractString,AbstractArray{AbstractString}}=nothing, seed::Union{Nothing,UInt32}=nothing, id::Int=1, init_radius=2.0, num_warmup::Int=1000, num_samples::Int=1000, metric::HMCMetric=DIAGONAL, init_inv_metric::Union{Nothing,Array{Float64}}=nothing, save_inv_metric::Bool=false, adapt::Bool=true, delta::Float64=0.8, gamma::Float64=0.05, kappa::Float64=0.75, t0::Int=10, init_buffer::Int=75, term_buffer::Int=50, window::Int=25, save_warmup::Bool=false, stepsize::Float64=1.0, stepsize_jitter::Float64=0.0, max_depth::Int=10, refresh::Int=0, num_threads::Int=-1)


Run Stan's No-U-Turn Sampler (NUTS) to sample from the posterior.
An in-depth explanation of the parameters can be found in the [Stan
documentation](https://mc-stan.org/docs/reference-manual/mcmc.html).

Returns StanOutput object with the draws, parameter names, and adapted stepsizes. If
`save_inv_metric` is true, the inverse metric is also returned.
"""
function sample(
    model::Model,
    data::AbstractString = "",
    ;
    num_chains::Int = 4,
    inits::Union{Nothing,AbstractString,AbstractVector{String}} = nothing,
    seed::Union{Nothing,UInt32} = nothing,
    id::Int = 1,
    init_radius = 2.0,
    num_warmup::Int = 1000,
    num_samples::Int = 1000,
    metric::HMCMetric = DIAGONAL,
    init_inv_metric::Union{Nothing,Array{Float64},Array{Float64,2},Array{Float64,3}} = nothing,
    save_inv_metric::Bool = false,
    adapt::Bool = true,
    delta::Float64 = 0.8,
    gamma::Float64 = 0.05,
    kappa::Float64 = 0.75,
    t0::Int = 10,
    init_buffer::Int = 75,
    term_buffer::Int = 50,
    window::Int = 25,
    save_warmup::Bool = false,
    stepsize::Float64 = 1.0,
    stepsize_jitter::Float64 = 0.0,
    max_depth::Int = 10,
    refresh::Int = 0,
    num_threads::Int = -1,
)
    if num_chains < 1
        error("num_chains must be at least 1")
    end
    if save_warmup && num_warmup < 0
        error("num_warmup must be non-negative")
    end
    if num_samples < 1
        error("num_samples must be at least 1")
    end

    if seed === nothing
        seed = rand(UInt32)
    end

    with_model(model, data, seed) do model_ptr
        free_params = num_free_params(model, model_ptr)

        param_names = cat(HMC_SAMPLER_VARIABLES, get_names(model, model_ptr), dims = 1)
        num_params = length(param_names)
        num_draws = num_samples + num_warmup * Int(save_warmup)
        out = zeros(Float64, num_params, num_draws, num_chains)

        if metric == DENSE
            metric_size = (free_params, free_params)
        else
            metric_size = (free_params,)
        end

        if isnothing(init_inv_metric)
            init_inv_metric = C_NULL
        else
            inv_metric_dims = size(init_inv_metric)
            if inv_metric_dims == metric_size
                init_inv_metric = repeat(
                    init_inv_metric,
                    outer = (ntuple(_ -> 1, length(inv_metric_dims))..., num_chains),
                )
            elseif inv_metric_dims == (metric_size..., num_chains)
                # good to go
            else
                with_chains = (metric_size..., num_chains)
                error(
                    "Invalid initial metric size. Expected a $metric_size or" *
                    " $with_chains matrix",
                )
            end
        end
        stepsize_out = C_NULL
        inv_metric_out = C_NULL
        if adapt
            stepsize_out = zeros(Float64, num_chains)
            if save_inv_metric
                inv_metric_out = zeros(Float64, metric_size..., num_chains)
            end
        end

        err = Ref{Ptr{Cvoid}}()
        return_code = @ccall $(dlsym(model.lib, :tinystan_sample))(
            model_ptr::Ptr{Cvoid},
            num_chains::Csize_t,
            encode_inits(model.sep, inits)::Cstring,
            seed::Cuint,
            id::Cuint,
            init_radius::Cdouble,
            num_warmup::Cint,
            num_samples::Cint,
            metric::HMCMetric,
            init_inv_metric::Ptr{Cdouble},
            adapt::Bool,
            delta::Cdouble,
            gamma::Cdouble,
            kappa::Cdouble,
            t0::Cdouble,
            init_buffer::Cint,
            term_buffer::Cint,
            window::Cint,
            save_warmup::Bool,
            stepsize::Cdouble,
            stepsize_jitter::Cdouble,
            max_depth::Cint,
            refresh::Cint,
            num_threads::Cint,
            out::Ref{Cdouble},
            length(out)::Csize_t,
            stepsize_out::Ptr{Cdouble},
            inv_metric_out::Ptr{Cdouble},
            err::Ref{Ptr{Cvoid}},
        )::Cint

        raise_for_error(model.lib, return_code, err)
        out = permutedims(out, (3, 2, 1))

        stepsizes = nothing
        inv_metric = nothing
        if adapt
            stepsizes = stepsize_out
            if save_inv_metric
                inv_metric = permutedims(
                    inv_metric_out,
                    range(length(size(inv_metric_out)), 1, step = -1),
                )
            end
        end

        return StanOutput{3}(param_names, out, stepsizes, inv_metric, nothing)
    end
end

"""
    pathfinder(model::Model, data::String=""; num_paths::Int=4, inits::Union{nothing,AbstractString,AbstractArray{AbstractString}}=nothing, seed::Union{Nothing,UInt32}=nothing, id::Int=1, init_radius=2.0, num_draws::Int=1000, max_history_size::Int=5, init_alpha::Float64=0.001, tol_obj::Float64=1e-12, tol_rel_obj::Float64=1e4, tol_grad::Float64=1e-8, tol_rel_grad::Float64=1e7, tol_param::Float64=1e-8, num_iterations::Int=1000, num_elbo_draws::Int=25, num_multi_draws::Int=1000, calculate_lp::Bool=true, psis_resample::Bool=true, refresh::Int=0, num_threads::Int=-1)

Run the Pathfinder algorithm to approximate the posterior.
See [Stan's documentation](https://mc-stan.org/docs/reference-manual/pathfinder.html)
for more information on the algorithm.

Returns StanOutput object with the draws, parameter names
"""
function pathfinder(
    model::Model,
    data::AbstractString = "",
    ;
    num_paths::Int = 4,
    inits::Union{Nothing,AbstractString,AbstractVector{String}} = nothing,
    seed::Union{UInt32,Nothing} = nothing,
    id::Int = 1,
    init_radius::Float64 = 2.0,
    num_draws::Int = 1000,
    max_history_size::Int = 5,
    init_alpha::Float64 = 0.001,
    tol_obj::Float64 = 1e-12,
    tol_rel_obj::Float64 = 1e4,
    tol_grad::Float64 = 1e-8,
    tol_rel_grad::Float64 = 1e7,
    tol_param::Float64 = 1e-8,
    num_iterations::Int = 1000,
    num_elbo_draws::Int = 25,
    num_multi_draws::Int = 1000,
    calculate_lp::Bool = true,
    psis_resample::Bool = true,
    refresh::Int = 0,
    num_threads::Int = -1,
)
    if num_draws < 1
        error("num_draws must be at least 1")
    end
    if num_paths < 1
        error("num_paths must be at least 1")
    end
    if num_multi_draws < 1
        error("num_multi_draws must be at least 1")
    end


    num_output = if calculate_lp && psis_resample
        num_multi_draws
    else
        num_draws * num_paths
    end

    if seed === nothing
        seed = rand(UInt32)
    end

    with_model(model, data, seed) do model_ptr
        free_params = num_free_params(model, model_ptr)
        if free_params == 0
            error("Model has no parameters")
        end

        param_names = cat(PATHFINDER_VARIABLES, get_names(model, model_ptr), dims = 1)
        num_params = length(param_names)
        out = zeros(Float64, num_params, num_output)

        err = Ref{Ptr{Cvoid}}()
        return_code = @ccall $(dlsym(model.lib, :tinystan_pathfinder))(
            model_ptr::Ptr{Cvoid},
            num_paths::Csize_t,
            encode_inits(model.sep, inits)::Cstring,
            seed::Cuint,
            id::Cuint,
            init_radius::Cdouble,
            num_draws::Cint,
            max_history_size::Cint,
            init_alpha::Cdouble,
            tol_obj::Cdouble,
            tol_rel_obj::Cdouble,
            tol_grad::Cdouble,
            tol_rel_grad::Cdouble,
            tol_param::Cdouble,
            num_iterations::Cint,
            num_elbo_draws::Cint,
            num_multi_draws::Cint,
            calculate_lp::Bool,
            psis_resample::Bool,
            refresh::Cint,
            num_threads::Cint,
            out::Ref{Cdouble},
            length(out)::Csize_t,
            err::Ref{Ptr{Cvoid}},
        )::Cint
        raise_for_error(model.lib, return_code, err)
        return StanOutput{2}(param_names, transpose(out), nothing, nothing, nothing)
    end
end

"""
    optimize(model::Model, data::String=""; init::Union{Nothing,AbstractString}=nothing, seed::Union{UInt32,Nothing}=nothing, id::Int=1, init_radius::Float64=2.0, algorithm::OptimizationAlgorithm=LBFGS, jacobian::Bool=false, num_iterations::Int=2000, max_history_size::Int=5, init_alpha::Float64=0.001, tol_obj::Float64=1e-12, tol_rel_obj::Float64=1e4, tol_grad::Float64=1e-8, tol_rel_grad::Float64=1e7, tol_param::Float64=1e-8, refresh::Int=0, num_threads::Int=-1)

Optimize the model parameters using the specified algorithm.

This will find either the maximum a posteriori (MAP) estimate
or the maximum likelihood estimate (MLE) of the model parameters,
depending on the value of the `jacobian` parameter.
Additional parameters can be found in the [Stan documentation](https://mc-stan.org/docs/reference-manual/optimization.html).

Returns StanOutput object with the draws, parameter names
"""
function optimize(
    model::Model,
    data::AbstractString = "",
    ;
    init::Union{Nothing,AbstractString} = nothing,
    seed::Union{UInt32,Nothing} = nothing,
    id::Int = 1,
    init_radius::Float64 = 2.0,
    algorithm::OptimizationAlgorithm = LBFGS,
    jacobian::Bool = false,
    num_iterations::Int = 2000,
    max_history_size::Int = 5,
    init_alpha::Float64 = 0.001,
    tol_obj::Float64 = 1e-12,
    tol_rel_obj::Float64 = 1e4,
    tol_grad::Float64 = 1e-8,
    tol_rel_grad::Float64 = 1e7,
    tol_param::Float64 = 1e-8,
    refresh::Int = 0,
    num_threads::Int = -1,
)
    if seed === nothing
        seed = rand(UInt32)
    end

    with_model(model, data, seed) do model_ptr
        param_names = cat(OPTIMIZE_VARIABLES, get_names(model, model_ptr), dims = 1)
        num_params = length(param_names)
        out = zeros(Float64, num_params)

        err = Ref{Ptr{Cvoid}}()
        return_code = @ccall $(dlsym(model.lib, :tinystan_optimize))(
            model_ptr::Ptr{Cvoid},
            if init === nothing
                C_NULL
            else
                init
            end::Cstring,
            seed::Cuint,
            id::Cuint,
            init_radius::Cdouble,
            algorithm::OptimizationAlgorithm,
            num_iterations::Cint,
            jacobian::Bool,
            max_history_size::Cint,
            init_alpha::Cdouble,
            tol_obj::Cdouble,
            tol_rel_obj::Cdouble,
            tol_grad::Cdouble,
            tol_rel_grad::Cdouble,
            tol_param::Cdouble,
            refresh::Cint,
            num_threads::Cint,
            out::Ref{Cdouble},
            length(out)::Csize_t,
            err::Ref{Ptr{Cvoid}},
        )::Cint
        raise_for_error(model.lib, return_code, err)
        return StanOutput{1}(param_names, out, nothing, nothing, nothing)
    end
end

"""
    laplace_sample(model::Model, mode::Union{AbstractString,Array{Float64}}, data::AbstractString=""; num_draws::Int=1000, jacobian::Bool=true, calculate_lp::Bool=true, save_hessian::Bool=false, seed::Union{UInt32,Nothing}=nothing, refresh::Int=0, num_threads::Int=-1)

Sample from the Laplace approximation of the posterior
centered at the provided mode. The mode can be either a JSON string
or an array of floats, often obtained from the [`optimize`](@ref)
function.

Returns StanOutput object with the draws, parameter names.
If `save_hessian` is true, the Hessian matrix is also returned.
"""
function laplace_sample(
    model::Model,
    mode::Union{AbstractString,Array{Float64}},
    data::AbstractString = "";
    num_draws::Int = 1000,
    jacobian::Bool = true,
    calculate_lp::Bool = true,
    save_hessian::Bool = false,
    seed::Union{UInt32,Nothing} = nothing,
    refresh::Int = 0,
    num_threads::Int = -1,
)
    if num_draws < 1
        error("num_draws must be at least 1")
    end
    if seed === nothing
        seed = rand(UInt32)
    end

    with_model(model, data, seed) do model_ptr
        required_params =  @ccall $(dlsym(model.lib, :tinystan_model_num_constrained_params_for_unconstraining))(
            model_ptr::Ptr{Cvoid},
        )::Cint

        if mode isa String
            mode_json = mode
            mode_array = C_NULL
        else
            mode_json = C_NULL
            mode_array = mode

            if length(mode_array) < required_params
                error(
                    "Mode array has incorrect length. Expected at least $required_params" *
                    " but got $(length(mode_array))",
                )
            end
        end

        param_names = cat(LAPLACE_VARIABLES, get_names(model, model_ptr), dims = 1)
        num_params = length(param_names)
        out = zeros(Float64, num_params, num_draws)

        if save_hessian
            free_params = num_free_params(model, model_ptr)
            hessian_out = zeros(Float64, free_params, free_params)
        else
            hessian_out = C_NULL
        end


        err = Ref{Ptr{Cvoid}}()
        return_code = @ccall $(dlsym(model.lib, :tinystan_laplace_sample))(
            model_ptr::Ptr{Cvoid},
            mode_array::Ptr{Cdouble},
            mode_json::Cstring,
            seed::Cuint,
            num_draws::Cint,
            jacobian::Bool,
            calculate_lp::Bool,
            refresh::Cint,
            num_threads::Cint,
            out::Ref{Cdouble},
            length(out)::Csize_t,
            hessian_out::Ptr{Cdouble},
            err::Ref{Ptr{Cvoid}},
        )::Cint
        raise_for_error(model.lib, return_code, err)

        hessian = nothing
        if save_hessian
            hessian = hessian_out
        end
        return StanOutput{2}(param_names, transpose(out), nothing, nothing, hessian)
    end
end
