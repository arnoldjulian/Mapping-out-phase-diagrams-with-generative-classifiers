# In this script, we analyze the 2D phase diagram of the classical anisotropic Ising model using the three schemes introduced in the main text
# Here, we approximate expected values with sample means. For routines which compute expected values exactly, as was done for Fig. 2 in the main text (as well as Fig. 2 in the SM), see script `run_main.jl`.

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
data_save_folder = "../../data/Ising/L="*string(L)*"/"

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
# and contains the distribution over the sufficient statistic (here, they take on 301 distinct/unique values) at each sampled point in parameter space,
# i.e., the relevant set of generative models
x_data = load(data_save_folder * "x_data.jld")["x_data"]

# set of distinct samples
samples = load(data_save_folder * "samples.jld")["samples"]
n_samples = length(samples)

# p_data of size length(γ1_range) x length(γ2_range) x number of tuning parameters and stores the value of all sampled points in parameter space
p_data = load(data_save_folder * "p_data.jld")["p_data"]

# load analytical reference phase boundary
ref_phase_boundary = load(data_save_folder * "ref_phase_boundary.jld")["ref_phase_boundary"]

# set number of samples to draw from generative model at each point in parameter space
n_samples = 100

##########
# scheme 1
##########

# define classes for scheme 1
n_classes = 5
class_data = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
    if class_indx == 1
        push!(class_data,[[10,10]])
    elseif class_indx == 2
        push!(class_data,[[10, length(γ2_range) - 10]])
    elseif class_indx == 3
        push!(class_data,[[length(γ1_range) - 10, 10]])
    elseif class_indx == 4
        push!(class_data,[[length(γ1_range) - 10, length(γ2_range) - 10]])
    else
        push!(class_data,[[Int(length(γ1_range) / 2), Int(length(γ2_range) / 2)]])
    end
end

# obtain indicator of scheme 1 based on set of generative models
_,_,I_1 = GCPT.run_scheme_1(x_data, class_data, [dγ1, dγ2],n_samples)

# plotting routine
function f(γ1, γ2, I_1)
    γ1_index = findall(x -> x == γ1, γ1_range[2:(end - 1)])[1]
    γ2_index = findall(x -> x == γ2, γ2_range[2:(end - 1)])[1]
    return I_1[γ1_index, γ2_index]
end

Plots.pyplot()
x = γ1_range[2:(end - 1)]
y = γ2_range[2:(end - 1)]

X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map((x, y) -> f(x, y, I_1), X, Y)

plt = contour(x, y, (x, y) -> f(x, y, I_1), fill = true, dpi = 300, color = :thermal)
xlabel!(L"J_{x}/k_{\mathrm{B}}T}")
ylabel!(L"J_{y}/k_{\mathrm{B}}T}")
plot!(ref_phase_boundary[:, 1, 1],
    ref_phase_boundary[:, 2, 1],
    color = "green",
    label = "analytical")
plot!(ref_phase_boundary[:, 1, 2],
    ref_phase_boundary[:, 2, 2],
    color = "green",
    label = false)
plot!(ref_phase_boundary[:, 1, 3],
    ref_phase_boundary[:, 2, 3],
    color = "green",
    label = false)
plot!(ref_phase_boundary[:, 1, 4],
    ref_phase_boundary[:, 2, 4],
    color = "green",
    label = false)
xlims!((γ1_range[2], γ1_range[end - 1]))
ylims!((γ2_range[2], γ2_range[end - 1]))
#savefig("./results_ising_scheme_1_w_sample_mean.png")

##########
# scheme 2
##########

# define l parameter to choose for scheme 2
# heuristic: choosing l to be small enough to capture only local variations but large enough to obtain a smooth signal by averaging over multiple points in parameter space
l_param = 1

# compute indicator of scheme 2 based on set of generative models
I_2 = GCPT.run_scheme_2(x_data,
    l_param,
    γ1_range,
    γ1_range_LBC,
    γ2_range,
    γ2_range_LBC,
    [dγ1, dγ2],n_samples)

# plotting routine
function f(γ1, γ2, I_2)
    γ1_index = findall(x -> x == γ1, γ1_range[2:(end - 1)])[1]
    γ2_index = findall(x -> x == γ2, γ2_range_LBC[2:(end - 1)])[1]
    return I_2[γ1_index, γ2_index]
end

Plots.pyplot()
x = γ1_range[2:(end - 1)]
y = γ2_range_LBC[2:(end - 1)]

X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map((x, y) -> f(x, y, I_2), X, Y)

plt = contourf(x,
    y,
    (x, y) -> f(x, y, I_2),
    fill = true,
    dpi = 300,
    color = :thermal,
    levels = 1000)
xlabel!(L"J_{x}/k_{\mathrm{B}}T}")
ylabel!(L"J_{y}/k_{\mathrm{B}}T}")
plot!(ref_phase_boundary[:, 1, 1],
    ref_phase_boundary[:, 2, 1],
    color = "green",
    label = "analytical")
plot!(ref_phase_boundary[:, 1, 2],
    ref_phase_boundary[:, 2, 2],
    color = "green",
    label = false)
plot!(ref_phase_boundary[:, 1, 3],
    ref_phase_boundary[:, 2, 3],
    color = "green",
    label = false)
plot!(ref_phase_boundary[:, 1, 4],
    ref_phase_boundary[:, 2, 4],
    color = "green",
    label = false)
xlims!((γ1_range[2], γ1_range[end - 1]))
ylims!((γ2_range[2], γ2_range[end - 1]))
#savefig("./results_ising_scheme_2_w_sample_mean.png")

##########
# scheme 3
##########

# compute indicator of scheme 3 based on set of generative models
# here the tuning parameters are estimated element-wise from linescan which decreases the bias of the corresponding estimator
I_3 = GCPT.run_scheme_3_linescans(x_data, p_data, [dγ1, dγ2],n_samples)

# alternatively, the tuning parameters can be estimated jointly from the entire parameter space (in case of the Ising model this leads to a qualitatively similar, but weaker indicator signal)
#_, _, I_3 = GCPT.run_scheme_3(x_data, p_data, [dγ1, dγ2], n_samples)

# plotting routine
function f(γ1, γ2, I_3)
    γ1_index = findall(x -> x == γ1, γ1_range[2:(end - 1)])[1]
    γ2_index = findall(x -> x == γ2, γ2_range[2:(end - 1)])[1]
    return I_3[γ1_index, γ2_index]
end

Plots.pyplot()
x = γ1_range[2:(end - 1)]
y = γ2_range[2:(end - 1)]

X = repeat(reshape(x, 1, :), length(y), 1)
Y = repeat(y, 1, length(x))
Z = map((x, y) -> f(x, y, I_3), X, Y)

plt = contourf(x,
    y,
    (x, y) -> f(x, y, I_3),
    fill = true,
    dpi = 300,
    color = :thermal,
    levels = 1000)
xlabel!(L"J_{x}/k_{\mathrm{B}}T}")
ylabel!(L"J_{y}/k_{\mathrm{B}}T}")
plot!(ref_phase_boundary[:, 1, 1],
    ref_phase_boundary[:, 2, 1],
    color = "green",
    label = "analytical")
plot!(ref_phase_boundary[:, 1, 2],
    ref_phase_boundary[:, 2, 2],
    color = "green",
    label = false)
plot!(ref_phase_boundary[:, 1, 3],
    ref_phase_boundary[:, 2, 3],
    color = "green",
    label = false)
plot!(ref_phase_boundary[:, 1, 4],
    ref_phase_boundary[:, 2, 4],
    color = "green",
    label = false)
xlims!((γ1_range[2], γ1_range[end - 1]))
ylims!((γ2_range[2], γ2_range[end - 1]))
#savefig("./results_ising_scheme_3_w_sample_mean.png")
