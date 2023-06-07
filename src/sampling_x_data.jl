
function get_probability(sampl,p,x_data)
    if length(p) == 1
        return x_data[p[1],sampl][1]
    else
	    return x_data[p[1],p[2],sampl][1]
    end
end

function get_sample(p,x_data)
    if length(p) == 1
        return StatsBase.sample(1:length(@view x_data[p[1],:]),Weights(@view x_data[p[1],:]),1)
    else
	    return StatsBase.sample(1:length(@view x_data[p[1],p[2],:]),Weights(@view x_data[p[1],p[2],:]),1)
    end
end
############################################################################################

function get_obs_SL_nD_f(sampl,class_data,dp,x_data)
	probs = zeros(eltype(dp[1]),length(class_data))

    if length(dp) == 1
        for i in 1:length(class_data)
            for p in class_data[i]
                probs[i] += get_probability(sampl, [p[1]],x_data)
            end
        end
    else
        for i in 1:length(class_data)
            for p in class_data[i]
                probs[i] += get_probability(sampl, [p[1],p[2]],x_data)
            end
        end
    end

	if sum(probs) > eps(eltype(dp[1]))
		return probs./sum(probs)
	else
		return zeros(eltype(p_range[1]),length(class_data))
	end
end

function get_mean_obs_SL_nD_f(n_samples,class_data,dp,x_data)
    param_space_dim = length(dp)

    if param_space_dim == 1
        mean_obs = zeros(eltype(dp[1]),(size(x_data)[1],1,length(class_data)))
        for j in 1:size(x_data)[1]
            for i in 1:n_samples
                mean_obs[j,1,:] .+= get_obs_SL_nD_f(get_sample([j],x_data),class_data,dp,x_data)
            end
        end
    
      elseif param_space_dim == 2
        mean_obs = zeros(eltype(dp[1]),(size(x_data)[1],size(x_data)[2],length(class_data)))
        for k in 1:size(x_data)[2]
            for j in 1:size(x_data)[1]
                for i in 1:n_samples
                    mean_obs[j,k,:] .+= get_obs_SL_nD_f(get_sample([j,k],x_data),class_data,dp,x_data)
                end
            end
        end

      else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
      end

	return mean_obs./n_samples
end

############################################################################################
function get_obs_SL_1D(sample,p_range,region_I,region_II,get_probability)
	prob_1 = zero(eltype(p_range[1]))
	prob_2 = zero(eltype(p_range[1]))

	for i in region_I
		prob_1 += get_probability(sample, i)
	end
	for i in region_II
		prob_2 += get_probability(sample, i)
	end

	if prob_1+prob_2 > eps(eltype(p_range[1]))
		return prob_1/(prob_1+prob_2)
	else
		return zero(eltype(p_range[1]))
	end
end

function get_mean_obs_SL_1D(n_samples,p_range,region_I,region_II,get_sample,get_probability)
	mean_obs = zeros(eltype(p_range[1]),length(p_range))
	for j in 1:length(p_range)
		for i in 1:n_samples
			mean_obs[j] += get_obs_SL(get_sample(j),p_range,region_I,region_II,get_probability)
		end
	end
	return mean_obs./n_samples
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

function get_obs_LBC_1D_f_TV(sampl,p_range,x_data,range_I,range_II)
    prob_1 = zero(eltype(p_range[1]))
    prob_2 = zero(eltype(p_range[1]))

    for i in range_I
        prob_1 += get_probability(sampl, [i], x_data)/length(range_I)
    end

    for i in range_II
        prob_2 += get_probability(sampl, [i], x_data)/length(range_II)
    end

	return minimum([prob_1/(prob_1+prob_2+eps(eltype(p_range[1]))),prob_2/(prob_1+prob_2+eps(eltype(p_range[1])))])
end

function get_mean_obs_LBC_1D_f_TV(n_samples,p_range,x_data,n_neighbors,p_tar_indx)
    range_I, range_II = get_LBC_ranges(p_range,n_neighbors,p_tar_indx)
	mean_obs = zero(eltype(p_range[1]))

	for j in range_I
		for i in 1:n_samples
			mean_obs += get_obs_LBC_1D_f_TV(get_sample([j],x_data),p_range,x_data,range_I, range_II)/(2*length(range_I))
		end
	end

    for j in range_II
		for i in 1:n_samples
			mean_obs += get_obs_LBC_1D_f_TV(get_sample([j],x_data),p_range,x_data,range_I, range_II)/(2*length(range_II))
		end
	end

    # 1-2*error
	return 1-2*mean_obs/n_samples
end
# could reuse samples drawn at different p_tar_indx values (becomes more crucial when range_I and range_II are large)

# function get_obs_LBC_1D_f_TV(sampl,p_range,x_data,n_neighbors,p)
# 	probs = map(i->get_probability(sampl, [i], x_data),1:length(p_range))
# 	prob_1 = zero(eltype(p_range[1]))
# 	prob_2 = zero(eltype(p_range[1]))
# 	obs = zeros(eltype(p_range[1]),length(p_range)-1)

#     for p_tar_indx in 1:length(p_range)-1
#         range_I, range_II = get_LBC_ranges(p_range,n_neighbors,p_tar_indx)

# 		prob_1 = sum((@view probs[range_I]),init=zero(eltype(p_range[1])))/length(range_I)
# 		prob_2 = sum((@view probs[range_II]),init=zero(eltype(p_range[1])))/length(range_II)
#         # importance sampling, i.e., reweighting by (length(p_range)/probs[p[1]])
# 		obs[p_tar_indx] = 2*minimum([prob_1/(prob_1+prob_2+eps(eltype(p_range[1]))),prob_2/(prob_1+prob_2+eps(eltype(p_range[1])))])*(1/probs[p[1]])*(prob_1+prob_2)/2
# 	end
# 	return obs
# end

# function get_mean_obs_LBC_1D_f_TV(n_samples,p_range,x_data,n_neighbors)
# 	mean_obs = zeros(eltype(p_range[1]),length(p_range)-1)
# 	for j in 1:length(p_range)
# 		for i in 1:n_samples
# 			mean_obs .+= get_obs_LBC_1D_f_TV(get_sample([j],x_data),p_range,x_data,n_neighbors,[j])
# 		end
# 	end
# 	return ones(eltype(p_range[1]),length(p_range)-1).-mean_obs./(length(p_range)*n_samples)
# end

############################################################################################

function get_obs_LBC_1D_f_old_ind(sampl,p_range,x_data)
	probs = map(i->get_probability(sampl, [i], x_data),1:length(p_range))
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

function get_mean_obs_LBC_1D_f_old_ind(n_samples,p_range,x_data)
	mean_obs = zeros(eltype(p_range[1]),length(p_range)+1)
	for j in 1:length(p_range)
		for i in 1:n_samples
			mean_obs .+= get_obs_LBC_1D_f_old_ind(get_sample([j],x_data),p_range,x_data)
		end
	end
	return ones(eltype(p_range[1]),length(p_range)+1).-mean_obs./(length(p_range)*n_samples)
end

###############################################################################################
function get_obs_LBC_1D(sample,p_range,get_probability)
	probs = map(i->get_probability(sample, i),1:length(p_range))
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

function get_mean_obs_LBC_1D(n_samples,p_range,get_sample,get_probability)
	mean_obs = zeros(eltype(p_range[1]),length(p_range)+1)
	for j in 1:length(p_range)
		for i in 1:n_samples
			mean_obs .+= get_obs_LBC(get_sample(j),p_range,get_probability)
		end
	end
	return ones(eltype(p_range[1]),length(p_range)+1).-mean_obs./(length(p_range)*n_samples)
end

############################################################################################
############################################################################################
############################################################################################
############################################################################################

function get_obs_PBM_nD_f(sampl,p_data,dp,x_data)
    param_space_dim = length(dp)
	obs = zeros(eltype(dp[1]),length(dp))
	norm = zero(eltype(dp[1]))

    if param_space_dim == 1
        for i in 1:size(p_data)[1]
            prob = get_probability(sampl, [i], x_data)
            obs .+= p_data[i,1]*prob
            norm += prob
        end
    else
        for j in 1:size(p_data)[2]
            for i in 1:size(p_data)[1]
                prob = get_probability(sampl, [i,j], x_data)
                obs .+= (@view p_data[i,j,:])*prob
                norm += prob
            end
        end
    end

	return obs./norm
end

function get_mean_obs_PBM_nD_f(n_samples,p_data,dp,x_data)
    param_space_dim = length(dp)
	mean_obs = zeros(eltype(dp[1]),size(p_data))

    if param_space_dim == 1
        for j in 1:size(p_data)[1]
            for i in 1:n_samples
                mean_obs[j,:] .+= get_obs_PBM_nD_f(get_sample([j],x_data),p_data,dp,x_data)
            end
        end
    
      elseif param_space_dim == 2
        for k in 1:size(p_data)[2]
            for j in 1:size(p_data)[1]
                for i in 1:n_samples
                    mean_obs[j,k,:] .+= get_obs_PBM_nD_f(get_sample([j,k],x_data),p_data,dp,x_data)
                end
            end
        end

      else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
      end

	return mean_obs./n_samples
end

###############################################################################################
function get_obs_PBM_1D(sample,p_range,get_probability)
	obs = zero(eltype(p_range[1]))
	norm = zero(eltype(p_range[1]))

	for i in 1:length(p_range)
		prob = get_probability(sample, i)
		obs += p_range[i]*prob
		norm += prob
	end

	return obs/norm
end

function get_mean_obs_PBM_1D(n_samples,p_range,get_sample,get_probability)
	mean_obs = zeros(eltype(p_range[1]),length(p_range))
	for j in 1:length(p_range)
		for i in 1:n_samples
			mean_obs[j] += get_obs_PBM(get_sample(j),p_range,get_probability)
		end
	end

	return mean_obs./n_samples
end