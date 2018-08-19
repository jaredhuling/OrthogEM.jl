__precompile__()

module oem


using Distributions
using LinearAlgebra, Arpack
using DataFrames, StaticArrays

export
    oem

include("utils.jl")
include("thresholds.jl")
include("oem_fit.jl")

end # module
