
struct StanOutput{D}
    names::Vector{String}
    draws::Array{Float64,D}
    stepsize::Union{Nothing,Vector{Float64}}
    inv_metric::Union{Nothing,Array{Float64,2},Array{Float64,3}}
    hessian::Union{Nothing,Array{Float64,2}}
end

function get_draws(output::StanOutput, name::String)
    colons = repeat([:], ndims(output.draws) - 1)
    return output.draws[colons..., output.names.==name]
end
