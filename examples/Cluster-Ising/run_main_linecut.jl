# In this script, we analyze a linecut through the 2D phase diagram of the cluster-Ising model using the three schemes introduced in the main text

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
ENV["GKSwstype"] = "nul"
using JLD
using ITensors
using ITensors.HDF5
using BenchmarkTools

# system size
L = 7

# set path to data folder
data_save_folder = "../../data/Cluster-Ising/L=" * string(L) * "/"

# define parameter ranges
γ1_min = 0.0f0
γ1_max = 1.25f0
γ2_min = -1.5f0
γ2_max = 1.5f0
dγ1 = 0.0125f0
dγ2 = 0.03f0
γ1_range = collect(γ1_min:dγ1:γ1_max)
γ2_range = collect(γ2_min:dγ2:γ2_max)
γ1_range_LBC = collect((γ1_min - dγ1 / 2):dγ1:(γ1_max + dγ1 / 2))
γ2_range_LBC = collect((γ2_min - dγ2 / 2):dγ2:(γ2_max + dγ2 / 2))

# In the following, we analyze a vertical linecut along γ_2 at γ_1 = 0.2 = γ1_range[17]
# We consider three distinct descriptions of the system: 1) a description in terms of numerically exact probability distribution underlying the system (x_data object),
# 2) a description in terms of wavefunctions obtained from exact diagonalization from which we sample, and 3) a description in terms of matrix-product-state (MPS) wavefunctions from which we sample
# Here, description 1 and 2 serve as comparisons. Note, however, that these two descriptions become intractable for larger systems.

# p_data of size length(γ1_range) x length(γ2_range) x number of tuning parameters and stores the value of all sampled points in parameter space
p_data = load(data_save_folder * "p_data.jld")["p_data"]

# description (1)

# x_data is of size length(γ2_range) x size of state space
# and contains the distribution over all measurement outcomes (which is 6^L = 279'936 for our measurement with Pauli-6 POVMs) at each sampled point in parameter space,
# i.e., the relevant set of generative models
x_data = load(data_save_folder * "x_data.jld")["x_data"]

# set of distinct samples
samples = load(data_save_folder * "samples.jld")["samples"]
n_samples = length(samples)

# description (2)

# wavefunc_data is a matrix of size length(γ2_range) x 2^L and contains all the corresponding ground-state wavefunctions
wavefunc_data = load(data_save_folder * "wavefunc_data.jld")["wavefunc_data"]

# description (3)

# MPS_data is a vector of size length(γ2_range) and contains all the corresponding MPS ground-state wavefunctions
f = h5open(data_save_folder * "MPS_h2_indx=1.h5", "r")
psi = read(f, "psi", ITensors.MPS)
close(f)

MPS_data = [psi]
for j in 2:length(γ2_range)
    f = h5open(data_save_folder * "MPS_h2_indx=" * string(j) * ".h5", "r")
    push!(MPS_data, read(f, "psi", ITensors.MPS))
    close(f)
end

# import sites object of MPS description
f = h5open(data_save_folder * "sites.h5", "r")
sites = read(f, "sites", ITensors.IndexSet)
close(f)

##########
# scheme 1
##########

# define classes for scheme 1
n_classes = 3
class_data = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
    if class_indx == 1
        push!(class_data, [[1]])
    elseif class_indx == 2
        push!(class_data, [[50]])
    elseif class_indx == 3
        push!(class_data, [[100]])
    end
end

# set number of samples to draw from generative models at each point in parameter space
n_samples = 1000

# obtain indicator of scheme 1 based on three distinct generative model descriptions
pred_1, I_1_classes, I_1 = GCPT.run_scheme_1(x_data, class_data, [dγ2])
pred_1_est_wf, I_1_classes_est_wf, I_1_est_wf = GCPT.run_scheme_1_wf(wavefunc_data,
    class_data,
    [dγ2],
    n_samples,
    L)
pred_1_est_MPS, I_1_classes_est_MPS, I_1_est_MPS = GCPT.run_scheme_1_MPS(MPS_data,
    class_data,
    [dγ2],
    n_samples,
    L,
    sites)

# plotting routine
plot(γ2_range,
    pred_1[:, 1],
    color = "black",
    label = "exact diagonalization, exact expectation",
    linewidth = 6,
    dpi = 300)
plot!(γ2_range, pred_1[:, 2], color = "black", label = false, linewidth = 6)
plot!(γ2_range, pred_1[:, 3], color = "black", label = false, linewidth = 6)

scatter!(γ2_range,
    pred_1_est_wf[:, 1, 1],
    color = "red",
    label = "exact diagonalization, sample mean")
scatter!(γ2_range, pred_1_est_wf[:, 1, 2], color = "blue", label = false)
scatter!(γ2_range, pred_1_est_wf[:, 1, 3], color = "green", label = false)

scatter!(γ2_range,
    pred_1_est_MPS[:, 1, 1],
    color = "red",
    label = "MPS representation, sample mean",
    markershape = :star5)
scatter!(γ2_range,
    pred_1_est_MPS[:, 1, 2],
    color = "blue",
    label = false,
    markershape = :star5)
scatter!(γ2_range,
    pred_1_est_MPS[:, 1, 3],
    color = "green",
    label = false,
    markershape = :star5)
xlabel!(L"h_{2}/J")
# here the three colors denote the three different classes
ylabel!(L"P(y|\gamma)")
#savefig("./predictions_cluster_ising_linecut_scheme_1.png")

plot(γ2_range[2:(end - 1)],
    I_1,
    color = "black",
    label = "exact diagonalization, exact expectation",
    linewidth = 6,
    dpi = 300)
scatter!(γ2_range[2:(end - 1)],
    I_1_est_wf,
    color = "blue",
    label = "exact diagonalization, sample mean",
    linewidth = 6)
scatter!(γ2_range[2:(end - 1)],
    I_1_est_MPS,
    color = "green",
    label = "MPS representation, sample mean",
    linewidth = 6)
xlabel!(L"h_{2}/J")
ylabel!(L"I_{1}")
#savefig("./indicator_cluster_ising_linecut_scheme_1.png")

##########
# scheme 2
##########

# define l parameter to choose for scheme 2
# heuristic: choosing l to be small enough to capture only local variations but large enough to obtain a smooth signal by averaging over multiple points in parameter space
l_param = 10

# set number of samples to draw from generative models at each point in parameter space
n_samples = 10

# obtain indicator of scheme 1 based on three distinct generative model descriptions
I_2 = GCPT.run_scheme_2(x_data,
    l_param,
    γ2_range,
    γ2_range_LBC,
    [dγ2])
I_2_est_wf = GCPT.run_scheme_2_wf(wavefunc_data,
    l_param,
    γ2_range,
    γ2_range_LBC,
    [dγ2],
    n_samples, L)
I_2_est_MPS = GCPT.run_scheme_2_MPS(MPS_data,
    l_param,
    γ2_range,
    γ2_range_LBC,
    [dγ2],
    n_samples, L, sites)

# plotting routine
plot(γ2_range_LBC[2:(end - 1)],
    I_2,
    color = "black",
    label = "exact diagonalization, exact expectation",
    linewidth = 6,
    dpi = 300,
    legend = :topright)
scatter!(γ2_range_LBC[2:(end - 1)],
    I_2_est_wf,
    color = "blue",
    label = "exact diagonalization, sample mean")
scatter!(γ2_range_LBC[2:(end - 1)],
    I_2_est_MPS,
    color = "green",
    label = "MPS representation, sample mean")
xlabel!(L"h_{2}/J")
ylabel!(L"I_{2}")
#savefig("./indicator_cluster_ising_linecut_scheme_2.png")

##########
# scheme 3
##########

# set number of samples to draw from generative models at each point in parameter space
n_samples = 100

# obtain indicator of scheme 1 based on three distinct generative model descriptions
pred_3, std_3, I_3 = GCPT.run_scheme_3(x_data, p_data, [dγ2])
pred_3_est_wf, std_3_est_wf, I_3_est_wf = GCPT.run_scheme_3_wf(wavefunc_data,
    p_data,
    [dγ2],
    n_samples,
    L)
pred_3_est_MPS, std_3_est_MPS, I_3_est_MPS = GCPT.run_scheme_3_MPS(MPS_data,
    p_data,
    [dγ2],
    n_samples,
    L,
    sites)

# plotting routine
plot(γ2_range,
    pred_3,
    color = "black",
    label = "exact diagonalization, exact expectation",
    linewidth = 6,
    dpi = 300,
    legend = :topleft)
scatter!(γ2_range,
    pred_3_est_wf,
    color = "blue",
    label = "exact diagonalization, sample mean")
scatter!(γ2_range,
    pred_3_est_MPS,
    color = "green",
    label = "MPS representation, sample mean")
xlabel!(L"h_{2}/J")
ylabel!(L"\hat{\gamma}(\gamma)")
#savefig("./prediction_cluster_ising_linecut_scheme_3.png")

plot(γ2_range,
    std_3,
    color = "black",
    label = "exact diagonalization, exact expectation",
    linewidth = 6,
    dpi = 300,
    legend = :topleft)
scatter!(γ2_range,
    std_3_est_wf,
    color = "blue",
    label = "exact diagonalization, sample mean")
scatter!(γ2_range,
    std_3_est_MPS,
    color = "green",
    label = "MPS representation, sample mean")
xlabel!(L"h_{2}/J")
ylabel!(L"\sigma(\gamma)")
xlims!((γ1_range[2], γ1_range[end - 1]))
#savefig("./standard_dev_cluster_ising_linecut_scheme_3.png")

plot(γ2_range[2:(end - 1)],
    I_3,
    color = "black",
    label = "exact diagonalization, exact expectation",
    linewidth = 6,
    dpi = 300,
    legend = :topright)
scatter!(γ2_range[2:(end - 1)],
    I_3_est_wf,
    color = "blue",
    label = "exact diagonalization, sample mean")
scatter!(γ2_range[2:(end - 1)],
    I_3_est_MPS,
    color = "green",
    label = "MPS representation, sample mean")
xlabel!(L"h_{2}/J")
ylabel!(L"I_{3}")
#savefig("./indicator_cluster_ising_linecut_scheme_3.png")

#to improve the results, increase the number of samples
