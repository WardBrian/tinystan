module FFIStan

export FFIStanModel, HMCMetric, pathfinder


@enum HMCMetric begin
    UNIT = 0
    DENSE = 1
    DIAG = 2
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

mutable struct FFIStanModel
    lib::Ptr{Nothing}
    const sep::Cchar

    function FFIStanModel(model::String)
        if endswith(model, ".stan")
            libname = model[1:end-5] * "_model.so"
            run(`make $libname`)
            lib = Libc.Libdl.dlopen(libname)
        else
            lib = Libc.Libdl.dlopen(model)
        end

        sep = ccall(Libc.Libdl.dlsym(lib, :ffistan_separator_char), Cchar, ())

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
        ccall(Libc.Libdl.dlsym(lib, :ffistan_free_stan_error), Cvoid, (Ptr{Cvoid},), err[])
        error(msg)
    end
end

function encode_inits(sep::Cchar, inits::Union{String,Vector{String},Nothing})
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

function with_model(f, model::FFIStanModel, data::String, seed::UInt32)
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

function get_names(model::FFIStanModel, model_ptr::Ptr{Cvoid})
    cstr = ccall(
        Libc.Libdl.dlsym(model.lib, :ffistan_model_param_names),
        Cstring,
        (Ptr{Cvoid},),
        model_ptr,
    )
    string.(split(unsafe_string(cstr), ','))
end

# algorithms

function pathfinder(
    model::FFIStanModel,
    data::String="",
    ;
    num_paths::Int=4,
    inits::Union{String,Vector{String},Nothing}=nothing,
    seed::Union{Int,Nothing}=nothing,
    id::Int=1,
    init_radius=2.0,
    num_draws=1000,
    max_history_size::Int=5,
    init_alpha=0.001,
    tol_obj=1e-12,
    tol_rel_obj=1e4,
    tol_grad=1e-8,
    tol_rel_grad=1e7,
    tol_param=1e-8,
    num_iterations::Int=1000,
    num_elbo_draws::Int=100,
    num_multi_draws::Int=1000,
    refresh::Int=0
)
    if num_paths < 1
        error("num_paths must be at least 1")
    end
    if num_draws < 1
        error("num_draws must be at least 1")
    end

    if seed === nothing
        seed = rand(UInt32)
    end

    with_model(model, data, seed) do model_ptr
        param_names = cat(PATHFINDER_VARIABLES, get_names(model, model_ptr), dims=1)
        num_params = length(param_names)
        out = zeros(Float64, num_draws * num_params)

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
                Ref{Cdouble},
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
            refresh,
            out,
            err,
        )
        raise_for_error(model.lib, return_code, err)
        return (param_names, transpose(reshape(out, num_params, num_draws)))

    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    ENV["STAN_NUM_THREADS"] = "-1"
    model = FFIStanModel("./bernoulli.stan")
    data = "bernoulli.data.json"

    param_names, draws = pathfinder(model, data)
    println(param_names)
    println(size(draws))
    using Statistics

    println(mean(draws, dims=1))
end

end
