
# TODO adapt for use with non-sampling methods
struct StanOutput
    names::Vector{String}
    draws::Array{Float64,3}  # (num_chains, num_draws, num_params)
    stepsize::Union{Nothing,Vector{Float64}}
    inv_metric::Union{Nothing,Array{Float64,2},Array{Float64,3}}
end
