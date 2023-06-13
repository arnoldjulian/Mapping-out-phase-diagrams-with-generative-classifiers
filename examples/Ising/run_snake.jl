# In this script, we analyze the 2D phase diagram of the classical anisotropic Ising model using the snake scheme

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
using Flux

# system size
L = 6

# set path to data folder
data_save_folder = "../../data/Ising/L="*string(L)*"/"

# define parameter ranges
p1_min = -1.475f0
p1_max = 1.475f0
p2_min = -1.475f0
p2_max = 1.475f0
dp1 = 0.05f0
dp2 = 0.05f0
p1_range = collect(p1_min:dp1:p1_max)
p2_range = collect(p2_min:dp2:p2_max)

rescale_x = p -> GCPT.rescale(p, p1_range)
rescale_y = p -> GCPT.rescale(p, p2_range)

inv_rescale_x = p -> GCPT.inv_rescale(p, p1_range)
inv_rescale_y = p -> GCPT.inv_rescale(p, p2_range)

p1_range_ren = rescale_x.(p1_range)
p2_range_ren = rescale_y.(p2_range)
dp1_ren = p1_range_ren[2] - p1_range_ren[1]
dp2_ren = p2_range_ren[2] - p2_range_ren[1]

# load data

# x_data is of size length(p1_range) x length(p2_range) x size of state space
# and contains the distribution over the sufficient statistic (here, they take on 301 distinct/unique values) at each sampled point in parameter space,
# i.e., the relevant set of generative models
x_data = load(data_save_folder * "x_data.jld")["x_data"]

# set of distinct samples
samples = load(data_save_folder * "samples.jld")["samples"]
n_samples = length(samples)

# p_data of size length(p1_range) x length(p2_range) x number of tuning parameters and stores the value of all sampled points in parameter space
p_data = load(data_save_folder * "p_data.jld")["p_data"]

# load analytical reference phase boundary
ref_phase_boundary = load(data_save_folder * "ref_phase_boundary.jld")["ref_phase_boundary"]

###########################################
# snake scheme using generative classifiers
###########################################

# set boundaries of the snake
boundaries = zeros(eltype(p1_min), (2, 2))
boundaries[1, :] = [dp1_ren, 1.0f0 - dp1_ren]
boundaries[2, :] = [dp2_ren, 1.0f0 - dp2_ren]

# define vertices/nodes of the snake 
function def_vertices(p1_range_ren, p2_range_ren)
    n_vertices = 20
    vertices = zeros(Float32, n_vertices, 2)
    for i in 1:n_vertices
        vertices[i, :] = [p1_range_ren[36 + i], p2_range_ren[59 - i]]
    end
    return vertices
end

# set hyperparameters
lr_int = 0.0001f0
lr_ext = 0.0005f0
l_param = 2
w0 = 5 * maximum([dp1_ren, dp2_ren])
w_end = minimum([dp1_ren, dp2_ren])
w_decay = 0.90f0
opt_internal = Adam(lr_int)
opt_external = Adam(lr_ext)
α = 0.002f0
β = 0.4f0
n_epochs = 400

# initialize snake
snake = GCPT.initialize_snake(def_vertices(p1_range_ren, p2_range_ren),
    PBC = false,
    FBC = false,
    boundaries,
    p1_range_ren,
    p2_range_ren,
    n_nodes = l_param * 2,
    α = α,
    β = β,
    w0 = w0,
    w_end = w_end,
    w_decay = w_decay,
    reshape = false,
    train_local = false,
    max_move = false,
    max_moves = [5 * dp1, 5 * dp2],
    treat_boundary = false)

# plot reference phase boundary and initial snake
plot(ref_phase_boundary[:, 1, 1],
    ref_phase_boundary[:, 2, 1],
    color = "green",
    label = "analytical",
    legend = :topleft)
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
scatter!(inv_rescale_x.(snake.vertices[:, 1]),
    inv_rescale_y.(snake.vertices[:, 2]),
    label = "Snake: Epoch 0",
    color = "black")

# train the snake (and plot its position)
colors = palette([:yellow, :red], n_epochs)
for epoch in 1:n_epochs
    GCPT.update_snake!(snake, x_data, n_samples, opt_internal, opt_external, epoch)
    if epoch % 50 == 0
        println(epoch)
        scatter!(inv_rescale_x.(snake.vertices[:, 1]),
            inv_rescale_y.(snake.vertices[:, 2]),
            label = "Snake: Epoch " * string(epoch),
            color = colors[epoch])
        plot!(inv_rescale_x.(snake.vertices[:, 1]),
            inv_rescale_y.(snake.vertices[:, 2]),
            label = false,
            color = colors[epoch])
    end
end
xlabel!(L"J_{x}/k_{\mathrm{B}}T}")
ylabel!(L"J_{y}/k_{\mathrm{B}}T}")
current()
#savefig("./results_ising_snake.png")
