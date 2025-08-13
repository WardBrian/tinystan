
"""
    StanOutput

A structure to hold the output from a Stan model run.
Always contains the names of the parameters and a "draws" array with
algorithm's output. Depending on the algorithm, it may also contain
`stepsize`, `inv_metric`, and `hessian` fields.
"""
struct StanOutput{D}
    names::Vector{String}
    draws::Array{Float64,D}
    stepsize::Union{Nothing,Vector{Float64}}
    inv_metric::Union{Nothing,Array{Float64,2},Array{Float64,3}}
    hessian::Union{Nothing,Array{Float64,2}}
end

"""
    get_draws(output::StanOutput, name::String)

Returns the draws for a specific parameter from the `StanOutput` object.
 """
function get_draws(output::StanOutput, name::String)
    namevec = output.names .== name
    if !any(namevec)
        error("Parameter '$name' not found in output.")
    end
    return selectdim(output.draws, ndims(output.draws), namevec)
end
