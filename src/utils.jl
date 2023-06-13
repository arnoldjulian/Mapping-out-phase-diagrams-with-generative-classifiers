# compute binary crossentropy loss
function crossentropy(p, l)
    # add epsilon perturbation for floating-point stability
    return -(l * log(p + eps(eltype(p))) + (1 - l) * log(1 - p + eps(eltype(p))))
end

#support functions for sampling from x_data object
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
