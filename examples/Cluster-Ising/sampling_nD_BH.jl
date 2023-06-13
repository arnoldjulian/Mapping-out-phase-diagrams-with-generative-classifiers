# activate project
cd(@__DIR__)
using Pkg;
Pkg.activate("../..");

# load packages
using GenClassifierPT
using DelimitedFiles
using LaTeXStrings
using Plots
using Random
using BenchmarkTools
ENV["GKSwstype"] = "nul"
using JLD
using StatsBase
using Test

# set path to save folder
data_folder = "/media/julian/Seagate Expansion Drive/2D_scans/BH/check_data/"
data_save_folder = "/home/julian/.julia/dev/ml_for_pt_2/data/BH/"

p1_range = vcat(readdlm(data_folder * "J_list.txt", eltype(Float32))...)
p2_range = vcat(readdlm(data_folder * "mu_list.txt", eltype(Float32))...)[2:end]

p1_min = minimum(p1_range)
p1_max = maximum(p1_range)
p2_min = minimum(p2_range)
p2_max = maximum(p2_range)
dp1 = p1_range[2] - p1_range[1]
dp2 = p2_range[2] - p2_range[1]
points = vcat(collect(Iterators.product(p1_range, p2_range))...)

p1_range_LBC = collect((p1_min - dp1 / 2):dp1:(p1_max + dp1 / 2))
p2_range_LBC = collect((p2_min - dp2 / 2):dp2:(p2_max + dp2 / 2))

p_data = load(data_save_folder * "p_data.jld")["p_data"]
x_data = load(data_save_folder * "x_data.jld")["x_data"]
samples = load(data_save_folder * "samples.jld")["samples"]

###################
n_classes = 4
class_data = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
    if class_indx == 1
        push!(class_data, [[1, 17], [1, 18]])
    elseif class_indx == 2
        push!(class_data, [[1, 50]])
    elseif class_indx == 3
        push!(class_data, [[1, 83]])
    elseif class_indx == 4
        push!(class_data, [[100, 17]])
    end
end

pred_SL_opt, indicator_SL_opt_classes, indicator_SL_opt_global = GCPT.run_scheme_1(x_data,
    class_data,
    [dp1, dp2])
pred_SL_opt_est, indicator_SL_opt_classes_est, indicator_SL_opt_global_est = GCPT.run_scheme_1(x_data,
    class_data_prob,
    [dp1, dp2],
    1000)

contourf(pred_SL_opt[:, :, 1]')
contourf(pred_SL_opt_est[:, :, 1]')

contourf(indicator_SL_opt_global[:, :, 1]')
contourf(indicator_SL_opt_global_est[:, :, 1]')

pred, std, I_3 = GCPT.run_scheme_3(x_data, p_data, [dp1, dp2])
pred_est, std_est, I_3_est = GCPT.run_scheme_3(x_data, p_data, [dp1, dp2], 10)

contourf(pred_PBM_opt[:, :, 1]')
contourf(pred_PBM_opt_est[:, :, 1]')

contourf(std_est')
contourf(I_3_est')

I_3 = GCPT.run_scheme_3_linescans(x_data, p_data, [dp1, dp2])
I_3_est = GCPT.run_scheme_3_linescans(x_data, p_data, [dp1, dp2], 1000)

contourf(I_3')
contourf(I_3_est')

pred, std, I_3 = GCPT.run_scheme_3(x_data[:, 17, :], p_data[:, 17, 1], [dp1])
pred_est, std_est, I_3_est = GCPT.run_scheme_3(x_data[:, 17, :],
    p_data[:, 17, 1],
    [dp1],
    100)

plot(p1_range, pred)
plot!(p1_range, pred_est)

plot(p1_range[2:(end - 1)], I_3)
plot!(p1_range[2:(end - 1)], I_3_est)

l_param = 10

I_2 = GCPT.run_scheme_2(x_data,
    l_param,
    p1_range,
    p1_range_LBC,
    p2_range,
    p2_range_LBC,
    [dp1, dp2])
I_2_est = GCPT.run_scheme_2(x_data,
    l_param,
    p1_range,
    p1_range_LBC,
    p2_range,
    p2_range_LBC,
    [dp1, dp2],
    10)

contourf(I_2')
contourf(I_2_est')

i = 17
n_samples = 1000
I_2 = GCPT.run_scheme_2(x_data[:, i, :], l_param, p1_range, p1_range_LBC, [dp1])
I_2_est = GCPT.run_scheme_2(x_data[:, i, :],
    l_param,
    p1_range,
    p1_range_LBC,
    [dp1],
    n_samples)

plot(p1_range_LBC[2:(end - 1)], I_2)
plot!(p1_range_LBC[2:(end - 1)], I_2_est)
