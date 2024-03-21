module FFIStan

export Model,
    HMCMetric,
    pathfinder,
    sample,
    optimize,
    OptimizationAlgorithm,
    api_version,
    compile_model,
    set_ffistan_path!

include("model.jl")
include("compile.jl")

end
