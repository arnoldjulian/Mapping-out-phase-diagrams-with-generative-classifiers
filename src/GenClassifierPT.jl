__precompile__()
module GenClassifierPT

# export package name as GCPT
export GCPT
const GCPT = GenClassifierPT

# load packages
using StaticArrays
using UnPack
using StatsBase
using DelimitedFiles
using Random
using Base.Threads
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using ITensors
using Flux

# include additional files
include("scheme_1.jl")
include("scheme_2.jl")
include("scheme_3.jl")
include("snake_scheme.jl")
include("utils.jl")
include("sampling_MPS.jl")
include("sampling_wf.jl")

end
