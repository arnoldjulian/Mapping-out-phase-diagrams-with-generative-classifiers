bt_x = [[1, 1] [1, -1]]./Float32(sqrt(2))
bt_y = adjoint([[1, 1im] [1, -1im]]./Float32(sqrt(2)))
bt_z = [[1, 0] [0, 1]]
const σ_list = [bt_x,bt_y,bt_z]

function get_probability_wf(s,p,wavefunc_data,L)
    if length(p) == 1
		prob = abs2.(s[2]*(@view wavefunc_data[p[1],:]))
        return prob[s[1][1]]/(3^L)
    else
		prob = abs2.(s[2]*(@view wavefunc_data[p[1],p[2],:]))
	    return prob[s[1][1]]/(3^L)
    end
end

function get_sample_wf!(p,wavefunc_data,L,sampl)	
	sampl[2] .= kron(MLP.σ_list[rand(1:3,L)]...)

    if length(p) == 1
		sampl[1] .= StatsBase.sample(1:length(@view wavefunc_data[1,:]),Weights(abs2.(sampl[2]*(@view wavefunc_data[p[1],:]))),1)[1]
        return sampl
    else
		sampl[1] .= StatsBase.sample(1:length(@view wavefunc_data[1,1,:]),Weights(abs2.(sampl[2]*(@view wavefunc_data[p[1],p[2],:]))),1)[1]
	    return sampl
    end
end

# function get_probability_wf(s,p,wavefunc_data,L)
#     if length(p) == 1
# 		prob = abs2.(s[2]*(@view wavefunc_data[p[1],:]))
#         return prob[s[1]]/(3^L)
#     else
# 		prob = abs2.(s[2]*(@view wavefunc_data[p[1],p[2],:]))
# 	    return prob[s[1]]/(3^L)
#     end
# end

# function get_sample_wf(p,wavefunc_data,L)	
# 	s = zeros(Int64,L+1)
# 	s[2:end] = rand(1:3,L)

# 	obs = MLP.σ_list[s[2]]
# 	for i in 2:L
# 		obs = kron(MLP.σ_list[s[i+1]],obs)
# 	end

#     if length(p) == 1
# 		sampl = sample(1:length(@view wavefunc_data[1,:]),Weights(abs2.(obs*(@view wavefunc_data[p[1],:]))),1)[1]
#         return (sampl,obs)
#     else
# 		sampl = sample(1:length(@view wavefunc_data[1,1,:]),Weights(abs2.(obs*(@view wavefunc_data[p[1],p[2],:]))),1)[1]
# 	    return (sampl,obs)
#     end
# end

###########

# function get_probability_wf(s,p,wavefunc_data,L)
# 	obs = σ_list[s[2]]
# 	for i in 2:L
# 		obs = kron(σ_list[s[i+1]],obs)
# 	end

#     if length(p) == 1
# 		prob = abs2.(obs*(@view wavefunc_data[p[1],:]))
#         return prob[s[1]]/(3^L)
#     else
# 		prob = abs2.(obs*(@view wavefunc_data[p[1],p[2],:]))
# 	    return prob[s[1]]/(3^L)
#     end
# end

# function get_sample_wf(p,wavefunc_data,L)	
# 	s = zeros(Int64,L+1)
# 	s[2:end] = rand(1:3,L)

# 	obs = σ_list[s[2]]
# 	for i in 2:L
# 		obs = kron(σ_list[s[i+1]],obs)
# 	end

#     if length(p) == 1
# 		s[1] = sample(1:length(@view wavefunc_data[1,:]),Weights(abs2.(obs*(@view wavefunc_data[p[1],:]))),1)[1]
#         return s
#     else
# 		s[1] = sample(1:length(@view wavefunc_data[1,1,:]),Weights(abs2.(obs*(@view wavefunc_data[p[1],p[2],:]))),1)[1]
# 	    return s
#     end
# end
# 1) make Pauli matrices global constants
# 2) maybe also return obs instead of integer sequence since one needs to reconstruct obs when evaluating the probability via get_probability
############################################################################################

function get_obs_SL_nD_f_wf(sampl,class_data,dp,wavefunc_data,L)
	probs = zeros(eltype(dp[1]),length(class_data))

    if length(dp) == 1
        for i in 1:length(class_data)
            for p in class_data[i]
                probs[i] += get_probability_wf(sampl, [p[1]],wavefunc_data,L)
            end
        end
    else
        for i in 1:length(class_data)
            for p in class_data[i]
                probs[i] += get_probability_wf(sampl, [p[1],p[2]],wavefunc_data,L)
            end
        end
    end

	if sum(probs) > eps(eltype(dp[1]))
		return probs./sum(probs)
	else
		return zeros(eltype(dp[1]),length(class_data))
	end
end

function get_mean_obs_SL_nD_f_wf(n_samples,class_data,dp,wavefunc_data,L)
    param_space_dim = length(dp)

    if param_space_dim == 1
        sampl = ([0],zeros(eltype(wavefunc_data[1,1]),(2^L,2^L)))
        mean_obs = zeros(eltype(dp[1]),(size(wavefunc_data)[1],1,length(class_data)))
        for j in 1:size(wavefunc_data)[1]
            for i in 1:n_samples
                mean_obs[j,1,:] .+= get_obs_SL_nD_f_wf(get_sample_wf!([j],wavefunc_data,L,sampl),class_data,dp,wavefunc_data,L)
            end
        end
    
      elseif param_space_dim == 2
        sampl = ([0],zeros(eltype(wavefunc_data[1,1,1]),(2^L,2^L)))
        mean_obs = zeros(eltype(dp[1]),(size(wavefunc_data)[1],size(wavefunc_data)[2],length(class_data)))
        for k in 1:size(wavefunc_data)[2]
            for j in 1:size(wavefunc_data)[1]
                for i in 1:n_samples
                    mean_obs[j,k,:] .+= get_obs_SL_nD_f_wf(get_sample_wf!([j,k],wavefunc_data,L,sampl),class_data,dp,wavefunc_data,L)
                end
            end
        end

      else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
      end

	return mean_obs./n_samples
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

function get_obs_LBC_1D_f_TV_wf(sampl,p_range,wavefunc_data,L,range_I,range_II)
    prob_1 = zero(eltype(p_range[1]))
    prob_2 = zero(eltype(p_range[1]))

    for i in range_I
        prob_1 += get_probability_wf(sampl, [i],wavefunc_data,L)/length(range_I)
    end

    for i in range_II
        prob_2 += get_probability_wf(sampl, [i],wavefunc_data,L)/length(range_II)
    end

	return minimum([prob_1/(prob_1+prob_2+eps(eltype(p_range[1]))),prob_2/(prob_1+prob_2+eps(eltype(p_range[1])))])
end

function get_mean_obs_LBC_1D_f_TV_wf(n_samples,p_range,wavefunc_data,L,n_neighbors,p_tar_indx)
    range_I, range_II = get_LBC_ranges(p_range,n_neighbors,p_tar_indx)
	mean_obs = zero(eltype(p_range[1]))
    sampl = ([0],zeros(eltype(wavefunc_data[1,1]),(2^L,2^L)))

	for j in range_I
		for i in 1:n_samples
			mean_obs += get_obs_LBC_1D_f_TV_wf(get_sample_wf!([j],wavefunc_data,L,sampl),p_range,wavefunc_data,L,range_I, range_II)/(2*length(range_I))
		end
	end

    for j in range_II
		for i in 1:n_samples
			mean_obs += get_obs_LBC_1D_f_TV_wf(get_sample_wf!([j],wavefunc_data,L,sampl),p_range,wavefunc_data,L,range_I, range_II)/(2*length(range_II))
		end
	end

    # 1-2*error
	return 1-2*mean_obs/n_samples
end
# could reuse samples drawn at different p_tar_indx values (becomes more crucial when range_I and range_II are large)

############################################################################################

function get_obs_LBC_1D_f_old_ind_wf(sampl,p_range,wavefunc_data,L)
	probs = map(i->get_probability_wf(sampl, [i], wavefunc_data,L),1:length(p_range))
	prob_1 = zero(eltype(p_range[1]))
	prob_2 = zero(eltype(p_range[1]))
	obs = zeros(eltype(p_range[1]),length(p_range)+1)

	for p_tar_indx in 1:length(p_range)+1
		prob_1 = sum((@view probs[1:p_tar_indx-1]),init=zero(eltype(p_range[1])))
		prob_2 = sum((@view probs[p_tar_indx:end]),init=zero(eltype(p_range[1])))
		obs[p_tar_indx] = minimum([prob_1/(prob_1+prob_2),prob_2/(prob_1+prob_2)])
	end

	return obs
end

function get_mean_obs_LBC_1D_f_old_ind_wf(n_samples,p_range,wavefunc_data,L)
	mean_obs = zeros(eltype(p_range[1]),length(p_range)+1)
    sampl = ([0],zeros(eltype(wavefunc_data[1,1]),(2^L,2^L)))
	for j in 1:length(p_range)
		for i in 1:n_samples
			mean_obs .+= get_obs_LBC_1D_f_old_ind_wf(get_sample_wf!([j],wavefunc_data,L,sampl),p_range,wavefunc_data,L)
		end
	end
	return ones(eltype(p_range[1]),length(p_range)+1).-mean_obs./(length(p_range)*n_samples)
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

function get_obs_PBM_nD_f_wf(sampl,p_data,dp,wavefunc_data,L)
    param_space_dim = length(dp)
	obs = zeros(eltype(dp[1]),length(dp))
	norm = zero(eltype(dp[1]))

    if param_space_dim == 1
        for i in 1:size(p_data)[1]
            prob = get_probability_wf(sampl, [i],wavefunc_data,L)
            obs .+= p_data[i,1]*prob
            norm += prob
        end
    else
        for j in 1:size(p_data)[2]
            for i in 1:size(p_data)[1]
                prob = get_probability_wf(sampl, [i,j],wavefunc_data,L)
                obs .+= (@view p_data[i,j,:])*prob
                norm += prob
            end
        end
    end

	return obs./norm
end

function get_mean_obs_PBM_nD_f_wf(n_samples,p_data,dp,wavefunc_data,L)
    param_space_dim = length(dp)
	mean_obs = zeros(eltype(dp[1]),size(p_data))

    if param_space_dim == 1
        sampl = ([0],zeros(eltype(wavefunc_data[1,1]),(2^L,2^L)))
        for j in 1:size(p_data)[1]
            for i in 1:n_samples
                mean_obs[j,:] .+= get_obs_PBM_nD_f_wf(get_sample_wf!([j],wavefunc_data,L,sampl),p_data,dp,wavefunc_data,L)
            end
        end
    
      elseif param_space_dim == 2
        sampl = ([0],zeros(eltype(wavefunc_data[1,1,1]),(2^L,2^L)))
        for k in 1:size(p_data)[2]
            for j in 1:size(p_data)[1]
                for i in 1:n_samples
                    mean_obs[j,k,:] .+= get_obs_PBM_nD_f_wf(get_sample_wf!([j,k],wavefunc_data,L,sampl),p_data,dp,wavefunc_data,L)
                end
            end
        end

      else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
      end

	return mean_obs./n_samples
end