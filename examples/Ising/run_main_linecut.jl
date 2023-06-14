# In this script, we analyze a linecut through the 2D phase diagram of the classical anisotropic Ising model using the three schemes introduced in the main text
# Here, we compute expected values exactly (similarly for Fig. 2 of the main text and Fig. 2 of the SM). For routines which approximate expected values with sample means, see script `run_main_2D_w_sample_mean.jl`.

# activate project
cd(@__DIR__)
using Pkg;
Pkg.activate("../..");

# load packages
using GenClassifierPT
using LaTeXStrings
using Plots
ENV["GKSwstype"] = "nul"
using JLD

# system size
L = 6

# set path to data folder
data_save_folder = "../../data/Ising/L=" * string(L) * "/"

# define parameter ranges
γ1_min = -1.475f0
γ1_max = 1.475f0
γ2_min = -1.475f0
γ2_max = 1.475f0
dγ1 = 0.05f0
dγ2 = 0.05f0
γ1_range = collect(γ1_min:dγ1:γ1_max)
γ2_range = collect(γ2_min:dγ2:γ2_max)
points = vcat(collect(Iterators.product(γ1_range, γ2_range))...)
γ1_range_LBC = collect((γ1_min - dγ1 / 2):dγ1:(γ1_max + dγ1 / 2))
γ2_range_LBC = collect((γ2_min - dγ2 / 2):dγ2:(γ2_max + dγ2 / 2))

# load data

# x_data is of size length(γ1_range) x length(γ2_range) x size of state space
# and contains the distribution over the sufficient statistic (here, they take on 39'571 distinct/unique values) at each sampled point in parameter space,
# i.e., the relevant set of generative models
x_data = load(data_save_folder * "x_data.jld")["x_data"]

# set of distinct samples
samples = load(data_save_folder * "samples.jld")["samples"]
n_samples = length(samples)

# p_data of size length(γ1_range) x length(γ2_range) x number of tuning parameters and stores the value of all sampled points in parameter space
p_data = load(data_save_folder * "p_data.jld")["p_data"]

# load analytical reference phase boundary
ref_phase_boundary = load(data_save_folder * "ref_phase_boundary.jld")["ref_phase_boundary"]

# in the following, we restrict ourselves to a horizontal linecut along γ_2 at γ_1 = -0.275
indx_lc = 20
γ1_range[indx_lc]

##########
# scheme 1
##########

# define classes for scheme 1
n_classes = 3
class_data = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
    if class_indx == 1
        push!(class_data, [[10]])
    elseif class_indx == 2
        push!(class_data, [[length(γ1_range) - 10]])
    else
        push!(class_data, [[Int(length(γ1_range) / 2)]])
    end
end

# obtain indicator of scheme 1 based on set of generative models
_, _, I_1 = GCPT.run_scheme_1(x_data[indx_lc, :, :], class_data, [dγ2])

# plotting routine
plot(γ2_range[2:(end - 1)], I_1, dpi = 300, label = false, color = "black")
xlabel!(L"J_{y}/k_{\mathrm{B}}T")
ylabel!(L"I_{1}")
xlims!((γ2_range[2], γ2_range[end - 1]))
vline!([ref_phase_boundary[indx_lc, 2, 3], ref_phase_boundary[indx_lc, 2, 4]],
    color = "red",
    label = "analytical")

#savefig("./results_ising_linecut_scheme_1.png")

##########
# scheme 2
##########

# define l parameter to choose for scheme 2
# heuristic: choosing l to be small enough to capture only local variations but large enough to obtain a smooth signal by averaging over multiple points in parameter space
l_param = 1

# compute indicator of scheme 2 based on set of generative models
I_2 = GCPT.run_scheme_2(x_data[indx_lc, :, :],
    l_param,
    γ2_range,
    γ2_range_LBC,
    [dγ2])

# plotting routine
plot(γ2_range_LBC[2:(end - 1)], I_2, dpi = 300, label = false, color = "black")
xlabel!(L"J_{y}/k_{\mathrm{B}}T")
ylabel!(L"I_{2}")
xlims!((γ2_range[2], γ2_range[end - 1]))
vline!([ref_phase_boundary[indx_lc, 2, 3], ref_phase_boundary[indx_lc, 2, 4]],
    color = "red",
    label = "analytical")
#savefig("./results_ising_linecut_scheme_2.png")

##########
# scheme 3
##########

# compute indicator of scheme 3 based on set of generative models
_, _, I_3 = GCPT.run_scheme_3(x_data[indx_lc, :, :], p_data[indx_lc, :, 2], [dγ2])

# plotting routine
plot(γ2_range[2:(end - 1)], I_3, dpi = 300, label = false, color = "black")
xlabel!(L"J_{y}/k_{\mathrm{B}}T")
ylabel!(L"I_{3}")
xlims!((γ2_range[2], γ2_range[end - 1]))
vline!([ref_phase_boundary[indx_lc, 2, 3], ref_phase_boundary[indx_lc, 2, 4]],
    color = "red",
    label = "analytical")
#savefig("./results_ising_linecut_scheme_3.png")
