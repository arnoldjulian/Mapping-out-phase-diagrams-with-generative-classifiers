function get_probability_MPS(s, p, MPS_data, sites, L)
    sampl = s[1]
    unit_seq = s[2]

    if length(p) == 1
        psi = deepcopy(MPS_data[p[1]])
    else
        psi = deepcopy(MPS_data[p[1], p[2]])
    end

    for j in 1:L
        if unit_seq[j] == 1
            opp = op("H", sites[j]) * psi[j]
            noprime!(opp)
            psi[j] = opp
        elseif unit_seq[j] == 2
            opp = op(adjoint([[1, 1im] [1, -1im]] ./ sqrt(2)), sites[j]) * psi[j]
            noprime!(opp)
            psi[j] = opp
        end
    end

    V = ITensors.ITensor(1)
    for i in 1:L
        V *= (psi[i] * state(sites[i], sampl[i]))
    end
    return abs2(ITensors.scalar(V))
end

function get_sample_MPS(p, MPS_data, sites, L)
    unit_seq = rand(1:3, L)

    if length(p) == 1
        psi = deepcopy(MPS_data[p[1]])
    else
        psi = deepcopy(MPS_data[p[1], p[2]])
    end

    for j in 1:L
        if unit_seq[j] == 1
            opp = op("H", sites[j]) * psi[j]
            noprime!(opp)
            psi[j] = opp
        elseif unit_seq[j] == 2
            opp = op(adjoint([[1, 1im] [1, -1im]] ./ sqrt(2)), sites[j]) * psi[j]
            noprime!(opp)
            psi[j] = opp
        end
    end
    return (ITensors.sample!(psi), unit_seq)
end
############################################################################################

function get_obs_SL_nD_f_MPS(sampl, class_data, dp, MPS_data, sites, L)
    probs = zeros(eltype(dp[1]), length(class_data))

    if length(dp) == 1
        for i in 1:length(class_data)
            for p in class_data[i]
                probs[i] += get_probability_MPS(sampl, [p[1]], MPS_data, sites, L)
            end
        end
    else
        for i in 1:length(class_data)
            for p in class_data[i]
                probs[i] += get_probability_MPS(sampl, [p[1], p[2]], MPS_data, sites, L)
            end
        end
    end

    if sum(probs) > eps(eltype(dp[1]))
        return probs ./ sum(probs)
    else
        return zeros(eltype(dp[1]), length(class_data))
    end
end

function get_mean_obs_SL_nD_f_MPS(n_samples, class_data, dp, MPS_data, sites, L)
    param_space_dim = length(dp)

    if param_space_dim == 1
        mean_obs = zeros(eltype(dp[1]), (size(MPS_data)[1], 1, length(class_data)))
        for j in 1:size(MPS_data)[1]
            for i in 1:n_samples
                mean_obs[j, 1, :] .+= get_obs_SL_nD_f_MPS(get_sample_MPS([j],
                        MPS_data,
                        sites,
                        L),
                    class_data,
                    dp,
                    MPS_data,
                    sites,
                    L)
            end
        end

    elseif param_space_dim == 2
        mean_obs = zeros(eltype(dp[1]),
            (size(MPS_data)[1], size(MPS_data)[2], length(class_data)))
        for k in 1:size(MPS_data)[2]
            for j in 1:size(MPS_data)[1]
                for i in 1:n_samples
                    mean_obs[j, k, :] .+= get_obs_SL_nD_f_MPS(get_sample_MPS([j, k],
                            MPS_data,
                            sites,
                            L),
                        class_data,
                        dp,
                        MPS_data,
                        sites,
                        L)
                end
            end
        end

    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    return mean_obs ./ n_samples
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

function get_obs_LBC_1D_f_TV_MPS(sampl, p_range, MPS_data, sites, L, range_I, range_II)
    prob_1 = zero(eltype(p_range[1]))
    prob_2 = zero(eltype(p_range[1]))

    for i in range_I
        prob_1 += get_probability_MPS(sampl, [i], MPS_data, sites, L) / length(range_I)
    end

    for i in range_II
        prob_2 += get_probability_MPS(sampl, [i], MPS_data, sites, L) / length(range_II)
    end

    return minimum([
        prob_1 / (prob_1 + prob_2 + eps(eltype(p_range[1]))),
        prob_2 / (prob_1 + prob_2 + eps(eltype(p_range[1]))),
    ])
end

function get_mean_obs_LBC_1D_f_TV_MPS(n_samples,
    p_range,
    MPS_data,
    sites,
    L,
    n_neighbors,
    p_tar_indx)
    range_I, range_II = get_LBC_ranges(p_range, n_neighbors, p_tar_indx)
    mean_obs = zero(eltype(p_range[1]))

    for j in range_I
        for i in 1:n_samples
            mean_obs += get_obs_LBC_1D_f_TV_MPS(get_sample_MPS([j], MPS_data, sites, L),
                p_range,
                MPS_data,
                sites,
                L,
                range_I,
                range_II) / (2 * length(range_I))
        end
    end

    for j in range_II
        for i in 1:n_samples
            mean_obs += get_obs_LBC_1D_f_TV_MPS(get_sample_MPS([j], MPS_data, sites, L),
                p_range,
                MPS_data,
                sites,
                L,
                range_I,
                range_II) / (2 * length(range_II))
        end
    end

    # 1-2*error
    return 1 - 2 * mean_obs / n_samples
end

############################################################################################

function get_obs_LBC_1D_f_old_ind_MPS(sampl, p_range, MPS_data, sites, L)
    probs = map(i -> get_probability_MPS(sampl, [i], MPS_data, sites, L), 1:length(p_range))
    prob_1 = zero(eltype(p_range[1]))
    prob_2 = zero(eltype(p_range[1]))
    obs = zeros(eltype(p_range[1]), length(p_range) + 1)

    for p_tar_indx in 1:(length(p_range) + 1)
        prob_1 = sum((@view probs[1:(p_tar_indx - 1)]), init = zero(eltype(p_range[1])))
        prob_2 = sum((@view probs[p_tar_indx:end]), init = zero(eltype(p_range[1])))
        obs[p_tar_indx] = minimum([prob_1 / (prob_1 + prob_2), prob_2 / (prob_1 + prob_2)])
    end

    return obs
end

function get_mean_obs_LBC_1D_f_old_ind_MPS(n_samples, p_range, MPS_data, sites, L)
    mean_obs = zeros(eltype(p_range[1]), length(p_range) + 1)
    for j in 1:length(p_range)
        for i in 1:n_samples
            mean_obs .+= get_obs_LBC_1D_f_old_ind_MPS(get_sample_MPS([j],
                    MPS_data,
                    sites,
                    L),
                p_range,
                MPS_data,
                sites,
                L)
        end
    end
    return ones(eltype(p_range[1]), length(p_range) + 1) .-
           mean_obs ./ (length(p_range) * n_samples)
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

function get_obs_PBM_nD_f_MPS(sampl, p_data, dp, MPS_data, sites, L)
    param_space_dim = length(dp)
    obs = zeros(eltype(dp[1]), length(dp))
    norm = zero(eltype(dp[1]))

    if param_space_dim == 1
        for i in 1:size(p_data)[1]
            prob = get_probability_MPS(sampl, [i], MPS_data, sites, L)
            obs .+= p_data[i, 1] * prob
            norm += prob
        end
    else
        for j in 1:size(p_data)[2]
            for i in 1:size(p_data)[1]
                prob = get_probability_MPS(sampl, [i, j], MPS_data, sites, L)
                obs .+= (@view p_data[i, j, :]) * prob
                norm += prob
            end
        end
    end

    return obs ./ norm
end

function get_mean_obs_PBM_nD_f_MPS(n_samples, p_data, dp, MPS_data, sites, L)
    param_space_dim = length(dp)
    mean_obs = zeros(eltype(dp[1]), size(p_data))

    if param_space_dim == 1
        for j in 1:size(p_data)[1]
            for i in 1:n_samples
                mean_obs[j, :] .+= get_obs_PBM_nD_f_MPS(get_sample_MPS([j],
                        MPS_data,
                        sites,
                        L),
                    p_data,
                    dp,
                    MPS_data,
                    sites,
                    L)
            end
        end

    elseif param_space_dim == 2
        for k in 1:size(p_data)[2]
            for j in 1:size(p_data)[1]
                for i in 1:n_samples
                    mean_obs[j, k, :] .+= get_obs_PBM_nD_f_MPS(get_sample_MPS([j, k],
                            MPS_data,
                            sites,
                            L),
                        p_data,
                        dp,
                        MPS_data,
                        sites,
                        L)
                end
            end
        end

    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    return mean_obs ./ n_samples
end

####################################################################################################

# function get_probability_MPS(sampl,p,MPS_data,sites,L)

#     if length(p) == 1
# 		V = ITensors.ITensor(1)
# 		for i in 1:L
# 			V *= (MPS_data[p[1]][i]*state(sites[i],sampl[i]))
# 		end
# 		return abs2(ITensors.scalar(V))
#     else
# 	    V = ITensors.ITensor(1)
# 		for i in 1:L
# 			V *= (MPS_data[p[1],p[2]][i]*state(sites[i],sampl[i]))
# 		end
# 		return abs2(ITensors.scalar(V))
#     end
# end

# function get_sample_MPS(p,MPS_data)
#     if length(p) == 1
#         return ITensors.sample(MPS_data[p[1]])
#     else
# 	    return ITensors.sample(MPS_data[p[1],p[2]])
#     end
# end
# ############################################################################################

# function get_obs_SL_nD_f_MPS(sampl,class_data,dp,MPS_data,sites,L)
# 	probs = zeros(eltype(dp[1]),length(class_data))

#     if length(dp) == 1
#         for i in 1:length(class_data)
#             for p in class_data[i]
#                 probs[i] += get_probability_MPS(sampl, [p[1]],MPS_data, sites, L)
#             end
#         end
#     else
#         for i in 1:length(class_data)
#             for p in class_data[i]
#                 probs[i] += get_probability_MPS(sampl, [p[1],p[2]],MPS_data, sites, L)
#             end
#         end
#     end

# 	if sum(probs) > eps(eltype(dp[1]))
# 		return probs./sum(probs)
# 	else
# 		return zeros(eltype(dp[1]),length(class_data))
# 	end
# end

# function get_mean_obs_SL_nD_f_MPS(n_samples,class_data,dp,MPS_data, sites, L)
#     param_space_dim = length(dp)

#     if param_space_dim == 1
#         mean_obs = zeros(eltype(dp[1]),(size(MPS_data)[1],1,length(class_data)))
#         for j in 1:size(MPS_data)[1]
#             for i in 1:n_samples
#                 mean_obs[j,1,:] .+= get_obs_SL_nD_f_MPS(get_sample_MPS([j],MPS_data),class_data,dp,MPS_data, sites, L)
#             end
#         end

#       elseif param_space_dim == 2
#         mean_obs = zeros(eltype(dp[1]),(size(MPS_data)[1],size(MPS_data)[2],length(class_data)))
#         for k in 1:size(MPS_data)[2]
#             for j in 1:size(MPS_data)[1]
#                 for i in 1:n_samples
#                     mean_obs[j,k,:] .+= get_obs_SL_nD_f_MPS(get_sample_MPS([j,k],MPS_data),class_data,dp,MPS_data, sites, L)
#                 end
#             end
#         end

#       else
#         error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
#       end

# 	return mean_obs./n_samples
# end

# ############################################################################################
# ############################################################################################
# ############################################################################################
# ############################################################################################

# function get_obs_LBC_1D_f_TV_MPS(sampl,p_range,MPS_data,sites,L,range_I,range_II)
#     prob_1 = zero(eltype(p_range[1]))
#     prob_2 = zero(eltype(p_range[1]))

#     for i in range_I
#         prob_1 += get_probability_MPS(sampl, [i],MPS_data,sites,L)/length(range_I)
#     end

#     for i in range_II
#         prob_2 += get_probability_MPS(sampl, [i],MPS_data,sites,L)/length(range_II)
#     end

# 	return minimum([prob_1/(prob_1+prob_2+eps(eltype(p_range[1]))),prob_2/(prob_1+prob_2+eps(eltype(p_range[1])))])
# end

# function get_mean_obs_LBC_1D_f_TV_MPS(n_samples,p_range,MPS_data,sites,L,n_neighbors,p_tar_indx)
#     range_I, range_II = get_LBC_ranges(p_range,n_neighbors,p_tar_indx)
# 	mean_obs = zero(eltype(p_range[1]))

# 	for j in range_I
# 		for i in 1:n_samples
# 			mean_obs += get_obs_LBC_1D_f_TV_MPS(get_sample_MPS([j],MPS_data),p_range,MPS_data,sites,L,range_I, range_II)/(2*length(range_I))
# 		end
# 	end

#     for j in range_II
# 		for i in 1:n_samples
# 			mean_obs += get_obs_LBC_1D_f_TV_MPS(get_sample_MPS([j],MPS_data),p_range,MPS_data,sites,L,range_I, range_II)/(2*length(range_II))
# 		end
# 	end

#     # 1-2*error
# 	return 1-2*mean_obs/n_samples
# end

# ############################################################################################

# function get_obs_LBC_1D_f_old_ind_MPS(sampl,p_range,MPS_data,sites,L)
# 	probs = map(i->get_probability_MPS(sampl, [i],MPS_data,sites,L),1:length(p_range))
# 	prob_1 = zero(eltype(p_range[1]))
# 	prob_2 = zero(eltype(p_range[1]))
# 	obs = zeros(eltype(p_range[1]),length(p_range)+1)

# 	for p_tar_indx in 1:length(p_range)+1
# 		prob_1 = sum((@view probs[1:p_tar_indx-1]),init=zero(eltype(p_range[1])))
# 		prob_2 = sum((@view probs[p_tar_indx:end]),init=zero(eltype(p_range[1])))
# 		obs[p_tar_indx] = minimum([prob_1/(prob_1+prob_2),prob_2/(prob_1+prob_2)])
# 	end

# 	return obs
# end

# function get_mean_obs_LBC_1D_f_old_ind_MPS(n_samples,p_range,MPS_data,sites,L)
# 	mean_obs = zeros(eltype(p_range[1]),length(p_range)+1)
# 	for j in 1:length(p_range)
# 		for i in 1:n_samples
# 			mean_obs .+= get_obs_LBC_1D_f_old_ind_MPS(get_sample_MPS([j],MPS_data),p_range,MPS_data,sites,L)
# 		end
# 	end
# 	return ones(eltype(p_range[1]),length(p_range)+1).-mean_obs./(length(p_range)*n_samples)
# end

# ############################################################################################
# ############################################################################################
# ############################################################################################
# ############################################################################################

# function get_obs_PBM_nD_f_MPS(sampl,p_data,dp,MPS_data,sites,L)
#     param_space_dim = length(dp)
# 	obs = zeros(eltype(dp[1]),length(dp))
# 	norm = zero(eltype(dp[1]))

#     if param_space_dim == 1
#         for i in 1:size(p_data)[1]
#             prob = get_probability_MPS(sampl, [i],MPS_data,sites,L)
#             obs .+= p_data[i,1]*prob
#             norm += prob
#         end
#     else
#         for j in 1:size(p_data)[2]
#             for i in 1:size(p_data)[1]
#                 prob = get_probability_MPS(sampl, [i,j],MPS_data,sites,L)
#                 obs .+= (@view p_data[i,j,:])*prob
#                 norm += prob
#             end
#         end
#     end

# 	return obs./norm
# end

# function get_mean_obs_PBM_nD_f_MPS(n_samples,p_data,dp,MPS_data,sites,L)
#     param_space_dim = length(dp)
# 	mean_obs = zeros(eltype(dp[1]),size(p_data))

#     if param_space_dim == 1
#         for j in 1:size(p_data)[1]
#             for i in 1:n_samples
#                 mean_obs[j,:] .+= get_obs_PBM_nD_f_MPS(get_sample_MPS([j],MPS_data),p_data,dp,MPS_data,sites,L)
#             end
#         end

#       elseif param_space_dim == 2
#         for k in 1:size(p_data)[2]
#             for j in 1:size(p_data)[1]
#                 for i in 1:n_samples
#                     mean_obs[j,k,:] .+= get_obs_PBM_nD_f_MPS(get_sample_MPS([j,k],MPS_data),p_data,dp,MPS_data,sites,L)
#                 end
#             end
#         end

#       else
#         error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
#       end

# 	return mean_obs./n_samples
# end
