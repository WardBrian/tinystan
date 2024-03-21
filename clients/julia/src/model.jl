
@enum HMCMetric begin
    UNIT = 0
    DENSE = 1
    DIAGONAL = 2
end

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

const PATHFINDER_VARIABLES = ["lp_approx__", "lp__"]

const OPTIMIZE_VARIABLES = ["lp__"]

const exceptions = [ErrorException, ArgumentError, _ -> InterruptException()]

mutable struct Model
    lib::Ptr{Nothing}
    const sep::Char

    function Model(
        model::String;
        stanc_args::AbstractVector{String} = String[],
        make_args::AbstractVector{String} = String[],
        warn::Bool = true,
    )
        if endswith(model, ".stan")
            libname = compile_model(model; stanc_args, make_args)
        else
            libname = model
        end

        if warn && in(abspath(libname), Libc.Libdl.dllist())
            @warn "Loading a shared object '" *
                  libname *
                  "' which is already loaded.\n" *
                  "If the file has changed since the last time it was loaded, this load may not update the library!"
        end

        windows_dll_path_setup()
        lib = Libc.Libdl.dlopen(libname)
        sep = Char(ccall(Libc.Libdl.dlsym(lib, :ffistan_separator_char), Cchar, ()))

        new(lib, sep)
    end

end


# helper functions
function raise_for_error(lib::Ptr{Nothing}, return_code::Cint, err::Ref{Ptr{Cvoid}})
    if return_code != 0
        if err[] == C_NULL
            error("Unknown error, function returned code $return_code")
        end
        cstr = ccall(
            Libc.Libdl.dlsym(lib, :ffistan_get_error_message),
            Cstring,
            (Ptr{Cvoid},),
            err[],
        )
        msg = unsafe_string(cstr)
        type = ccall(
            Libc.Libdl.dlsym(lib, :ffistan_get_error_type),
            Cint,
            (Ptr{Cvoid},),
            err[],
        )
        ccall(Libc.Libdl.dlsym(lib, :ffistan_free_stan_error), Cvoid, (Ptr{Cvoid},), err[])
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
    model_ptr = ccall(
        Libc.Libdl.dlsym(model.lib, :ffistan_create_model),
        Ptr{Cvoid},
        (Cstring, Cuint, Ref{Ptr{Cvoid}}),
        data,
        seed,
        err,
    )

    raise_for_error(model.lib, Int32(model_ptr == C_NULL), err)

    try
        return f(model_ptr)
    finally
        ccall(
            Libc.Libdl.dlsym(model.lib, :ffistan_destroy_model),
            Cvoid,
            (Ptr{Cvoid},),
            model_ptr,
        )
    end
end

function num_free_params(model::Model, model_ptr::Ptr{Cvoid})
    Int(
        ccall(
            Libc.Libdl.dlsym(model.lib, :ffistan_model_num_free_params),
            Csize_t,
            (Ptr{Cvoid},),
            model_ptr,
        ),
    )
end

function get_names(model::Model, model_ptr::Ptr{Cvoid})
    cstr = ccall(
        Libc.Libdl.dlsym(model.lib, :ffistan_model_param_names),
        Cstring,
        (Ptr{Cvoid},),
        model_ptr,
    )
    str = unsafe_string(cstr)
    if isempty(str)
        return String[]
    end
    string.(split(unsafe_string(cstr), ','))
end

function api_version(model::Model)
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    cstr = ccall(
        Libc.Libdl.dlsym(model.lib, :ffistan_api_version),
        Cvoid,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
        major,
        minor,
        patch,
    )
    (major[], minor[], patch[])
end

# algorithms

function sample(
    model::Model,
    data::String = "",
    ;
    num_chains::Int = 4,
    inits = nothing,
    seed::Union{Nothing,UInt32} = nothing,
    id::Int = 1,
    init_radius = 2.0,
    num_warmup::Int = 1000,
    num_samples::Int = 1000,
    metric::HMCMetric = DIAGONAL,
    init_inv_metric::Union{Nothing,Array{Float64}} = nothing,
    save_metric::Bool = false,
    adapt = true,
    delta = 0.8,
    gamma = 0.05,
    kappa = 0.75,
    t0::Int = 10,
    init_buffer::Int = 75,
    term_buffer::Int = 50,
    window::Int = 25,
    save_warmup::Bool = false,
    stepsize = 1.0,
    stepsize_jitter = 0.0,
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
        if free_params == 0
            error("Model has no parameters to sample")
        end

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

        if save_metric
            metric_out = zeros(Float64, metric_size..., num_chains)
        else
            metric_out = C_NULL
        end

        err = Ref{Ptr{Cvoid}}()
        return_code = ccall(
            Libc.Libdl.dlsym(model.lib, :ffistan_sample),
            Cint,
            (
                Ptr{Cvoid},
                Csize_t,
                Cstring,
                Cuint,
                Cuint,
                Cdouble,
                Cint,
                Cint,
                HMCMetric,
                Ptr{Cdouble},
                Cint, # really bool
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cuint,
                Cuint,
                Cuint,
                Cint, # really bool
                Cdouble,
                Cdouble,
                Cint,
                Cint,
                Cint,
                Ref{Cdouble},
                Csize_t,
                Ptr{Cdouble},
                Ref{Ptr{Cvoid}},
            ),
            model_ptr,
            num_chains,
            encode_inits(model.sep, inits),
            seed,
            id,
            init_radius,
            num_warmup,
            num_samples,
            metric,
            init_inv_metric,
            Int32(adapt),
            delta,
            gamma,
            kappa,
            t0,
            init_buffer,
            term_buffer,
            window,
            Int32(save_warmup),
            stepsize,
            stepsize_jitter,
            max_depth,
            refresh,
            num_threads,
            out,
            length(out),
            metric_out,
            err,
        )
        raise_for_error(model.lib, return_code, err)
        out = permutedims(out, (3, 2, 1))
        if save_metric
            metric_out =
                permutedims(metric_out, range(length(size(metric_out)), 1, step = -1))
            return (param_names, out, metric_out)
        end
        return (param_names, out)
    end
end

function pathfinder(
    model::Model,
    data::String = "",
    ;
    num_paths::Int = 4,
    inits::Union{String,Vector{String},Nothing} = nothing,
    seed::Union{UInt32,Nothing} = nothing,
    id::Int = 1,
    init_radius = 2.0,
    num_draws = 1000,
    max_history_size::Int = 5,
    init_alpha = 0.001,
    tol_obj = 1e-12,
    tol_rel_obj = 1e4,
    tol_grad = 1e-8,
    tol_rel_grad = 1e7,
    tol_param = 1e-8,
    num_iterations::Int = 1000,
    num_elbo_draws::Int = 100,
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
        return_code = ccall(
            Libc.Libdl.dlsym(model.lib, :ffistan_pathfinder),
            Cint,
            (
                Ptr{Cvoid},
                Csize_t,
                Cstring,
                Cuint,
                Cuint,
                Cdouble,
                Cint,
                Cint,
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cint,
                Cint,
                Cint,
                Cint,
                Cint,
                Cint,
                Cint,
                Ref{Cdouble},
                Csize_t,
                Ref{Ptr{Cvoid}},
            ),
            model_ptr,
            num_paths,
            encode_inits(model.sep, inits),
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
            Int32(calculate_lp),
            Int32(psis_resample),
            refresh,
            num_threads,
            out,
            length(out),
            err,
        )
        raise_for_error(model.lib, return_code, err)
        return (param_names, transpose(out))

    end
end

function optimize(
    model::Model,
    data::String = "",
    ;
    init::Union{String,Nothing} = nothing,
    seed::Union{UInt32,Nothing} = nothing,
    id::Int = 1,
    init_radius = 2.0,
    algorithm::OptimizationAlgorithm = LBFGS,
    jacobian::Bool = false,
    num_iterations::Int = 2000,
    max_history_size::Int = 5,
    init_alpha = 0.001,
    tol_obj = 1e-12,
    tol_rel_obj = 1e4,
    tol_grad = 1e-8,
    tol_rel_grad = 1e7,
    tol_param = 1e-8,
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
        return_code = ccall(
            Libc.Libdl.dlsym(model.lib, :ffistan_optimize),
            Cint,
            (
                Ptr{Cvoid},
                Cstring,
                Cuint,
                Cuint,
                Cdouble,
                Cint, # really enum
                Cint,
                Cint, # really bool
                Cint,
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cdouble,
                Cint,
                Cint,
                Ref{Cdouble},
                Csize_t,
                Ref{Ptr{Cvoid}},
            ),
            model_ptr,
            if init === nothing
                C_NULL
            else
                init
            end,
            seed,
            id,
            init_radius,
            algorithm,
            num_iterations,
            Int32(jacobian),
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
            length(out),
            err,
        )
        raise_for_error(model.lib, return_code, err)
        return (param_names, out)
    end
end
