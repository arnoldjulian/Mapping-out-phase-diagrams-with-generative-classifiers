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
using Combinatorics
using Permutations
using ITensors
using ITensors.HDF5

# set path to save folder
save_folder = "./results/"

data_folder = "/media/julian/Seagate Expansion Drive/2D_scans/SPT_POVM/data/L=3/run=1/"
data_save_folder = "../../data/SPT_POVM/L=3/run=1/"

L = 3
p1_range = vcat(readdlm(data_folder * "h1_list.txt", eltype(Float32))...)
p2_range = vcat(readdlm(data_folder * "h2_list.txt", eltype(Float32))...)

p1_min = minimum(p1_range)
p1_max = maximum(p1_range)
p2_min = minimum(p2_range)
p2_max = maximum(p2_range)
dp1 = p1_range[2] - p1_range[1]
dp2 = p2_range[2] - p2_range[1]
points = vcat(collect(Iterators.product(p1_range, p2_range))...)

p1_range_LBC = collect((p1_min - dp1 / 2):dp1:(p1_max + dp1 / 2))
p2_range_LBC = collect((p2_min - dp2 / 2):dp2:(p2_max + dp2 / 2))

p_data_z = load(data_save_folder * "p_data.jld")["p_data"]
x_data_z = load(data_save_folder * "x_data.jld")["x_data"]
samples_z = load(data_save_folder * "samples.jld")["samples"]

prob = vcat(readdlm(data_folder * "/probs_Pauli6_h2=" * string(0) * ".txt",
    eltype(Float64))...)
x_data = zeros(eltype(Float32), length(p2_range), length(prob))
p_data = zeros(eltype(Float32), length(p2_range), 1)

for j in 1:length(p2_range)
    x_data[j, :] = vcat(readdlm(data_folder * "/probs_Pauli6_h2=" * string(j - 1) * ".txt",
        eltype(Float64))...)
    p_data[j, :] = [p2_range[j]]
end
samples = collect(1:length(prob))
n_samples = length(samples)
n_samples = length(samples)

t = vcat(readdlm(data_folder * "/gs_real_h1=0_h2=0.txt", eltype(Float32))...)
wavefunc_data = zeros(ComplexF32, (length(p1_range), length(p2_range), length(t)))
for j in 1:length(p2_range)
    for i in 1:length(p1_range)
        wavefunc_data[i, j, :] .= vcat(readdlm(data_folder * "/gs_real_h1=" *
                                               string(i - 1) * "_h2=" * string(j - 1) *
                                               ".txt",
            eltype(Float32))...) .+
                                  vcat(readdlm(data_folder * "/gs_imag_h1=" *
                                               string(i - 1) * "_h2=" * string(j - 1) *
                                               ".txt",
            eltype(Float32))...) * im
    end
end

wavefunc_str = split.(vcat(readdlm(data_folder * "/strings.txt", String)...), "")
wavefunc_int = zeros(Int64, (2^L, L))
for i in 1:(2^L)
    wavefunc_int[i, :] = parse.(Int64, wavefunc_str[i])
end

####
f = h5open(data_folder * "sites.h5", "r")
sites = read(f, "sites", ITensors.IndexSet)
close(f)

f = h5open(data_folder * "MPS_h1=16_h2=0.h5", "r")
psi = read(f, "psi", ITensors.MPS)
close(f)

MPS_data = [psi]
for j in 1:(length(p2_range) - 1)
    f = h5open(data_folder * "MPS_h1=16_h2=" * string(j) * ".h5", "r")
    push!(MPS_data, read(f, "psi", ITensors.MPS))
    close(f)
end
###################
n_classes = 3
class_data = zeros(eltype(n_classes), length(p2_range), n_classes)
class_data_prob = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
    if class_indx == 1
        class_data[1, class_indx] = 1
        push!(class_data_prob, [[1]])
    elseif class_indx == 2
        class_data[50, class_indx] = 1
        push!(class_data_prob, [[50]])
    elseif class_indx == 3
        class_data[100, class_indx] = 1
        push!(class_data_prob, [[100]])
    end
end

# pred_SL_opt, indicator_SL_opt_classes, indicator_SL_opt_global = MLP.get_indicators_SL_analytical_nD(x_data, class_data, [dp2])
# pred_SL_opt_est = MLP.get_mean_obs_SL_nD_f_wf(1000,class_data_prob,[dp2],wavefunc_data[17,:,:],3)
# pred_SL_opt_est_MPS = MLP.get_mean_obs_SL_nD_f_MPS(1000,class_data_prob,[dp2],MPS_data,sites,L)

pred_SL_opt, indicator_SL_opt_classes, indicator_SL_opt_global = MLP.get_indicators_SL_analytical_nD(abs2.(wavefunc_data[17,
        :,
        :]),
    class_data,
    [dp2])
pred_SL_opt_est = MLP.get_mean_obs_SL_nD_f(1000,
    class_data_prob,
    [dp2],
    abs2.(wavefunc_data[17, :, :]))
pred_SL_opt_est_MPS = MLP.get_mean_obs_SL_nD_f_MPS(1000,
    class_data_prob,
    [dp2],
    MPS_data,
    sites,
    L)

plot(p2_range,
    pred_SL_opt[:, 1],
    color = "black",
    label = "ED Exact",
    linewidth = 6,
    dpi = 300)
plot!(p2_range, pred_SL_opt[:, 2], color = "black", label = false, linewidth = 6)
plot!(p2_range, pred_SL_opt[:, 3], color = "black", label = false, linewidth = 6)

scatter!(p2_range, pred_SL_opt_est[:, 1, 1], color = "red", label = "ED Sampling")
scatter!(p2_range, pred_SL_opt_est[:, 1, 2], color = "blue", label = false)
scatter!(p2_range, pred_SL_opt_est[:, 1, 3], color = "green", label = false)

scatter!(p2_range,
    pred_SL_opt_est_MPS[:, 1, 1],
    color = "red",
    label = "MPS Sampling",
    markershape = :star5)
scatter!(p2_range,
    pred_SL_opt_est_MPS[:, 1, 2],
    color = "blue",
    label = false,
    markershape = :star5)
scatter!(p2_range,
    pred_SL_opt_est_MPS[:, 1, 3],
    color = "green",
    label = false,
    markershape = :star5)

# pred_PBM_opt, _, divergence_opt, loss_opt = MLP.get_indicators_PBM_analytical_nD(x_data, p_data, [dp2])
# pred_PBM_opt_est = MLP.get_mean_obs_PBM_nD_f_wf(1000,p_data,[dp2],wavefunc_data[17,:,:],L)
# pred_PBM_opt_est_MPS = MLP.get_mean_obs_PBM_nD_f_MPS(1000,p_data,[dp2],MPS_data,sites,L)

pred_PBM_opt, _, divergence_opt, loss_opt = MLP.get_indicators_PBM_analytical_nD(abs2.(wavefunc_data[17,
        :,
        :]),
    p_data,
    [dp2])
pred_PBM_opt_est = MLP.get_mean_obs_PBM_nD_f(1000,
    p_data,
    [dp2],
    abs2.(wavefunc_data[17, :, :]))
pred_PBM_opt_est_MPS = MLP.get_mean_obs_PBM_nD_f_MPS(100, p_data, [dp2], MPS_data, sites, L)

plot(p2_range,
    pred_PBM_opt,
    color = "black",
    label = "ED Exact",
    linewidth = 6,
    dpi = 300,
    legend = :topleft)
scatter!(p2_range, pred_PBM_opt_est, color = "blue", label = "ED Sampling")
scatter!(p2_range, pred_PBM_opt_est_MPS, color = "green", label = "MPS Sampling")

# ind_LBC_opt, _ = MLP.get_indicators_LBC_analytical(x_data',p2_range)
# ind_LBC_opt_est = MLP.get_mean_obs_LBC_1D_f_old_ind_wf(100,p2_range,wavefunc_data[17,:,:],L)

ind_LBC_opt, _ = MLP.get_indicators_LBC_analytical(abs2.(wavefunc_data[17, :, :])',
    p2_range)
ind_LBC_opt_est = MLP.get_mean_obs_LBC_1D_f_old_ind(100,
    p2_range,
    abs2.(wavefunc_data[17, :, :]))
ind_LBC_opt_est_MPS = MLP.get_mean_obs_LBC_1D_f_old_ind_MPS(100,
    p2_range,
    MPS_data,
    sites,
    L)

plot(p2_range_LBC,
    ind_LBC_opt,
    color = "black",
    label = "ED Exact",
    linewidth = 6,
    dpi = 300,
    legend = :topleft)
scatter!(p2_range_LBC, ind_LBC_opt_est, color = "blue", label = "ED Sampling")
scatter!(p2_range_LBC, ind_LBC_opt_est_MPS, color = "green", label = "MPS Sampling")

n_neighbors = 4
# _, ind_LBC_opt_TV = MLP.get_indicators_LBC_analytical_unbiased_range(x_data', p2_range, n_neighbors)
# ind_LBC_opt_TV_est = map(p_tar_indx->MLP.get_mean_obs_LBC_1D_f_TV_wf(1000,p2_range,wavefunc_data[17,:,:],L,n_neighbors,p_tar_indx),1:length(p1_range)-1)

_, ind_LBC_opt_TV = MLP.get_indicators_LBC_analytical_unbiased_range(abs2.(wavefunc_data[17,
        :,
        :])',
    p2_range,
    n_neighbors)
ind_LBC_opt_TV_est = map(p_tar_indx -> MLP.get_mean_obs_LBC_1D_f_TV(100,
        p2_range,
        abs2.(wavefunc_data[17, :, :]),
        n_neighbors,
        p_tar_indx),
    1:(length(p2_range) - 1))
ind_LBC_opt_TV_est_MPS = map(p_tar_indx -> MLP.get_mean_obs_LBC_1D_f_TV_MPS(100,
        p2_range,
        MPS_data,
        sites,
        L,
        n_neighbors,
        p_tar_indx),
    1:(length(p2_range) - 1))

plot(p2_range_LBC[2:(end - 1)],
    ind_LBC_opt_TV,
    color = "black",
    label = "ED Exact",
    linewidth = 6,
    dpi = 300,
    legend = :topright)
scatter!(p2_range_LBC[2:(end - 1)],
    ind_LBC_opt_TV_est,
    color = "blue",
    label = "ED Sampling")
scatter!(p2_range_LBC[2:(end - 1)],
    ind_LBC_opt_TV_est_MPS,
    color = "green",
    label = "MPS Sampling")
