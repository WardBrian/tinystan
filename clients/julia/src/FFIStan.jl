module FFIStan

export FFIStanModel,
    HMCMetric, pathfinder, sample, optimize, OptimizationAlgorithm, api_version

include("model.jl")
include("compile.jl")

function __init__()
    # On Windows, we may need to add TBB to %PATH%
    if Sys.iswindows()
        try
            run(pipeline(`where.exe tbb.dll`, stdout = devnull, stderr = devnull))
        catch
            # add TBB to %PATH%
            ENV["PATH"] =
                joinpath(get_ffistan_path(), "stan", "lib", "stan_math", "lib", "tbb") *
                ";" *
                ENV["PATH"]
        end
    end
end

end
