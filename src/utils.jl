# compute binary crossentropy loss
function crossentropy(p, l)
    # add epsilon perturbation for floating-point stability
    return -(l * log(p + eps(eltype(p))) + (1 - l) * log(1 - p + eps(eltype(p))))
end

#support functions for sampling from x_data object
function get_probability(sampl, γ, x_data)
    if length(γ) == 1
        return x_data[γ[1], sampl][1]
    elseif length(γ) == 2
        return x_data[γ[1], γ[2], sampl][1]
    else
        error("Parameter spaces with dimension > 2 are currently not supported.")
    end
end

function get_sample(γ, x_data)
    if length(γ) == 1
        return StatsBase.sample(1:length(@view x_data[γ[1], :]),
            Weights(@view x_data[γ[1], :]),
            1)
    elseif length(γ) == 2
        return StatsBase.sample(1:length(@view x_data[γ[1], γ[2], :]),
            Weights(@view x_data[γ[1], γ[2], :]),
            1)
    else
        error("Parameter spaces with dimension > 2 are currently not supported.")
    end
end

#support functions for sampling from wavefunctions obtained from exact diagonalization
bt_x = [[1, 1] [1, -1]] ./ Float32(sqrt(2))
bt_y = adjoint([[1, 1im] [1, -1im]] ./ Float32(sqrt(2)))
bt_z = [[1, 0] [0, 1]]
const σ_list = [bt_x, bt_y, bt_z]

function get_probability_wf(s, γ, wavefunc_data, L)
    if length(γ) == 1
        prob = abs2.(s[2] * (@view wavefunc_data[γ[1], :]))
        return prob[s[1][1]] / (3^L)
    else
        prob = abs2.(s[2] * (@view wavefunc_data[γ[1], γ[2], :]))
        return prob[s[1][1]] / (3^L)
    end
end

function get_sample_wf!(γ, wavefunc_data, L, sampl)
    sampl[2] .= kron(GCPT.σ_list[rand(1:3, L)]...)

    if length(γ) == 1
        sampl[1] .= StatsBase.sample(1:length(@view wavefunc_data[1, :]),
            Weights(abs2.(sampl[2] * (@view wavefunc_data[γ[1], :]))),
            1)[1]
        return sampl
    else
        sampl[1] .= StatsBase.sample(1:length(@view wavefunc_data[1, 1, :]),
            Weights(abs2.(sampl[2] * (@view wavefunc_data[γ[1], γ[2], :]))),
            1)[1]
        return sampl
    end
end

#support functions for sampling from MPS wavefunctions
function get_probability_MPS(s, γ, MPS_data, L, sites)
    sampl = s[1]
    unit_seq = s[2]

    if length(γ) == 1
        psi = deepcopy(MPS_data[γ[1]])
    else
        psi = deepcopy(MPS_data[γ[1], γ[2]])
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

function get_sample_MPS(γ, MPS_data, L, sites)
    unit_seq = rand(1:3, L)

    if length(γ) == 1
        psi = deepcopy(MPS_data[γ[1]])
    else
        psi = deepcopy(MPS_data[γ[1], γ[2]])
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
