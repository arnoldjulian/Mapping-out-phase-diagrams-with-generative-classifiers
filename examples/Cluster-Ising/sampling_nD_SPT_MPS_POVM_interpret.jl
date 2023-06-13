# make your scripts automatically re-activate your project
cd(@__DIR__)
using Pkg; Pkg.activate("../..")

# load packages
using GenClassifierPT
using DelimitedFiles
using LaTeXStrings
using Plots
using Random
using BenchmarkTools
ENV["GKSwstype"]="nul"
using JLD
using StatsBase
using Test
using Combinatorics
using Permutations
using ITensors
using ITensors.HDF5

# set path to save folder
save_folder = "./results/"

L = 7
data_folder = "/media/julian/Seagate Expansion Drive/2D_scans/SPT_POVM/data/L="*string(L)*"/run=1/"
data_save_folder = "../../data/SPT_POVM/L="*string(L)*"/run=1/"

p1_range = vcat(readdlm(data_folder*"h1_list.txt",eltype(Float32))...)
p2_range = vcat(readdlm(data_folder*"h2_list.txt",eltype(Float32))...)

p1_min = minimum(p1_range)
p1_max = maximum(p1_range)
p2_min = minimum(p2_range)
p2_max = maximum(p2_range)
dp1 = p1_range[2]-p1_range[1]
dp2 = p2_range[2]-p2_range[1]
points = vcat(collect(Iterators.product(p1_range, p2_range))...)

p1_range_LBC = collect(p1_min-dp1/2:dp1:p1_max+dp1/2)
p2_range_LBC = collect(p2_min-dp2/2:dp2:p2_max+dp2/2)

t = vcat(readdlm(data_folder*"/gs_real_h1=0_h2=0.txt",eltype(Float32))...) 
wavefunc_data = zeros(ComplexF32,(length(p1_range),length(p2_range),length(t)))
for j in 1:length(p2_range)
	for i in 1:length(p1_range)
		wavefunc_data[i,j,:] .= vcat(readdlm(data_folder*"/gs_real_h1="*string(i-1)*"_h2="*string(j-1)*".txt",eltype(Float64))...) .+ vcat(readdlm(data_folder*"/gs_imag_h1="*string(i-1)*"_h2="*string(j-1)*".txt",eltype(Float64))...)*im 
	end
end

wavefunc_str = split.(vcat(readdlm(data_folder*"/strings.txt",String)...),"")
wavefunc_int = zeros(Int64,(2^L,L))
for i in 1:2^L
	wavefunc_int[i,:] = parse.(Int64,wavefunc_str[i])
end
wavefunc_int = 2*wavefunc_int.-1

n_classes = 3
class_data = zeros(eltype(n_classes),length(p2_range),n_classes)
class_data_prob = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
	if class_indx == 1
		class_data[1,class_indx] = 1
		data = [[1]]
		for i in 2:15
			push!(data,[i])
		end
		push!(class_data_prob,data)
	elseif class_indx == 2
		class_data[50,class_indx] = 1
		data = [[40]]
		for i in 41:60
			push!(data,[i])
		end
		push!(class_data_prob,data)
	elseif class_indx == 3
		class_data[100,class_indx] = 1
		data = [[85]]
		for i in 86:length(p2_range)
			push!(data,[i])
		end
		push!(class_data_prob,data)
	end
end

function interpret_wf(p,wavefunc_data,L,wavefunc_int)
	seq = rand(1:3,L)

	unit = kron(MLP.σ_list[seq]...)
	sampl = StatsBase.sample(1:length(@view wavefunc_data[1,:]),Weights(abs2.(unit*(@view wavefunc_data[p[1],:]))),1)[1]
    return seq.*wavefunc_int[sampl,:], (abs2.(unit*(@view wavefunc_data[p[1],:])))[sampl]/(3^L), sampl, unit
end

seqss = []
probss = []
n_samples = 5000
for n in 1:n_samples 
	#for p_indx in 85:length(p2_range)
	#for p_indx in 1:15
	for p_indx in 20:80
		seq, prob, sampl, unit = interpret_wf([p_indx],wavefunc_data[17,:,:],L,wavefunc_int)
		prob_SL = MLP.get_obs_SL_nD_f_wf((sampl, unit),class_data_prob,[dp2],wavefunc_data[17,:,:],L)[2]
		push!(seqss,seq)
		push!(probss,prob*prob_SL)
	end
end

sortpermm = reverse(sortperm(probss))
probss[sortpermm]
uniques = unique(seqss[sortpermm])
top_op = abs.(uniques[1])
top_op_sign = sign.(uniques[1])

s_x = [[0,1] [1,0]]
s_y = [[1, 1im] [-1im, 1]]
s_z = [[1, 0] [0, -1]]
ss_list = [s_x,s_y,s_z]
obs = kron(ss_list[top_op]...)

obs_list = map(p->real(adjoint(wavefunc_data[17,p,:])*obs*wavefunc_data[17,p,:]),1:length(p2_range))
plot(p2_range,obs_list)






###################
n_classes = 3
class_data = zeros(eltype(n_classes),length(p2_range),n_classes)
class_data_prob = Vector{Vector{Int64}}[]
for class_indx in 1:n_classes
	if class_indx == 1
		class_data[1,class_indx] = 1
		push!(class_data_prob,[[1]])
	elseif class_indx == 2
		class_data[50,class_indx] = 1
		push!(class_data_prob,[[50]])
	elseif class_indx == 3
		class_data[100,class_indx] = 1
		push!(class_data_prob,[[100]])
	end
end

pred_SL_opt, indicator_SL_opt_classes, indicator_SL_opt_global = MLP.get_indicators_SL_analytical_nD(x_data, class_data, [dp2])
pred_SL_opt_est = MLP.get_mean_obs_SL_nD_f_wf(100,class_data_prob,[dp2],wavefunc_data[17,:,:],L)
pred_SL_opt_est_MPS = MLP.get_mean_obs_SL_nD_f_MPS(100,class_data_prob,[dp2],MPS_data,sites,L)

plot(p2_range,pred_SL_opt[:,1],color="black",label="ED Exact",linewidth=6,dpi=300)
plot!(p2_range,pred_SL_opt[:,2],color="black",label=false,linewidth=6)
plot!(p2_range,pred_SL_opt[:,3],color="black",label=false,linewidth=6)

scatter!(p2_range,pred_SL_opt_est[:,1,1],color="red",label="ED Sampling")
scatter!(p2_range,pred_SL_opt_est[:,1,2],color="blue",label=false)
scatter!(p2_range,pred_SL_opt_est[:,1,3],color="green",label=false)

scatter!(p2_range,pred_SL_opt_est_MPS[:,1,1],color="red",label="MPS Sampling",markershape=:star5)
scatter!(p2_range,pred_SL_opt_est_MPS[:,1,2],color="blue",label=false,markershape=:star5)
scatter!(p2_range,pred_SL_opt_est_MPS[:,1,3],color="green",label=false,markershape=:star5)

pred_PBM_opt, _, divergence_opt, loss_opt = MLP.get_indicators_PBM_analytical_nD(x_data, p_data, [dp2])
pred_PBM_opt_est = MLP.get_mean_obs_PBM_nD_f_wf(1000,p_data,[dp2],wavefunc_data[17,:,:],L)
pred_PBM_opt_est_MPS = MLP.get_mean_obs_PBM_nD_f_MPS(100,p_data,[dp2],MPS_data,sites,L)

plot(p2_range,pred_PBM_opt,color="black",label="ED Exact",linewidth=6,dpi=300,legend=:topleft)
scatter!(p2_range,pred_PBM_opt_est,color="blue",label="ED Sampling")
scatter!(p2_range,pred_PBM_opt_est_MPS,color="green",label="MPS Sampling")

ind_LBC_opt, _ = MLP.get_indicators_LBC_analytical(x_data',p2_range)
ind_LBC_opt_est = MLP.get_mean_obs_LBC_1D_f_old_ind_wf(1,p2_range,wavefunc_data[17,:,:],L)
ind_LBC_opt_est_MPS = MLP.get_mean_obs_LBC_1D_f_old_ind_MPS(1,p2_range,MPS_data,sites,L)

plot(p2_range_LBC,ind_LBC_opt,color="black",label="ED Exact",linewidth=6,dpi=300,legend=:topleft)
scatter!(p2_range_LBC,ind_LBC_opt_est,color="blue",label="ED Sampling")
scatter!(p2_range_LBC,ind_LBC_opt_est_MPS,color="green",label="MPS Sampling")

n_neighbors = 4
_, ind_LBC_opt_TV = MLP.get_indicators_LBC_analytical_unbiased_range(x_data', p2_range, n_neighbors)
ind_LBC_opt_TV_est = map(p_tar_indx->MLP.get_mean_obs_LBC_1D_f_TV_wf(10,p2_range,wavefunc_data[17,:,:],L,n_neighbors,p_tar_indx),1:length(p1_range)-1)
ind_LBC_opt_TV_est_MPS = map(p_tar_indx->MLP.get_mean_obs_LBC_1D_f_TV_MPS(10,p2_range,MPS_data,sites,L,n_neighbors,p_tar_indx),1:length(p2_range)-1)

plot(p2_range_LBC[2:end-1],ind_LBC_opt_TV,color="black",label="ED Exact",linewidth=6,dpi=300,legend=:topright)
scatter!(p2_range_LBC[2:end-1],ind_LBC_opt_TV_est,color="blue",label="ED Sampling")
scatter!(p2_range_LBC[2:end-1],ind_LBC_opt_TV_est_MPS,color="green",label="MPS Sampling")