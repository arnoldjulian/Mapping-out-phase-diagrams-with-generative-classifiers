function get_scheme_2_ranges(p_range, n_neighbors, p_indx)
    range_1 = eltype(p_indx)[]
    range_2 = eltype(p_indx)[]
    for i in (p_indx - n_neighbors + 1):p_indx
        if i > 0
            push!(range_1, i)
        end
    end

    for i in (p_indx + 1):(p_indx + n_neighbors)
        if i < length(p_range) + 1
            push!(range_2, i)
        end
    end

    return range_1, range_2
end

# scheme 2 with expected values being evaluated exactly
function run_scheme_2(x_data, l_param, γ1_range, γ1_range_LBC, γ2_range, γ2_range_LBC, dγ)
    I_2_x = zeros(eltype(dγ[1]), length(γ1_range), length(γ2_range_LBC) - 2)
    for i in 1:length(γ1_range)
        I_2_x[i, :] = run_scheme_2_loc(x_data[i, :, :]', γ2_range, l_param)
    end

    I_2_y = zeros(eltype(dγ[1]), length(γ1_range_LBC) - 2, length(γ2_range))
    for i in 1:length(γ2_range)
        I_2_y[:, i] = run_scheme_2_loc(x_data[:, i, :]', γ1_range, l_param)
    end

    I_2 = sqrt.((I_2_x .^ 2)[1:(end - 1), :] .+ (I_2_y .^ 2)[:, 1:(end - 1)])
    return I_2
end

function run_scheme_2(x_data, l_param, γ_range, γ_range_LBC, dγ)
    I_2 = run_scheme_2_loc(x_data', γ_range, l_param)

    return I_2
end

function run_scheme_2_loc(data, γ_range, l_param)
    I_2_loc = zeros(eltype(γ_range[1]), length(γ_range) - 1)

    for γ_tar_indx in 1:(length(γ_range) - 1)
        range_I, range_II = get_scheme_2_ranges(γ_range, Int(l_param), γ_tar_indx)
        I_2_loc[γ_tar_indx] = run_scheme_2_fixed_bipartition(data,
            γ_range,
            range_I,
            range_II)
    end

    return I_2_loc
end

function run_scheme_2_fixed_bipartition(data, p_range, range_I, range_II)
    p1 = sum((@view data[:, range_I]), dims = 2, init = zero(eltype(p_range[1])))[:, 1] ./
         length(range_I)
    p2 = sum((@view data[:, range_II]), dims = 2, init = zero(eltype(p_range[1])))[:, 1] ./
         length(range_II)
    pred_opt = p1 ./ (p1 .+ p2 .+ eps(eltype(p_range[1])))

    prob_eff = ((sum((@view data[:, range_I]), dims = 2) ./ length(range_I)) .+
                (sum((@view data[:, range_II]), dims = 2) ./ length(range_II))) ./ 2
    error = sum(prob_eff .*
                (min.(pred_opt, ones(eltype(p_range[1]), length(pred_opt)) .- pred_opt)))

    return 1 - 2 * error
end

# scheme 2 with expectation values being approximated with sample mean, where n_samples denotes the number of samples drawn at each point in parameter space
function run_scheme_2(x_data,
    l_param,
    γ1_range,
    γ1_range_LBC,
    γ2_range,
    γ2_range_LBC,
    dγ,
    n_samples)
    I_2_x = zeros(eltype(dγ[1]), length(γ1_range), length(γ2_range_LBC) - 2)
    for i in 1:length(γ1_range)
        I_2_x[i, :] = run_scheme_2_loc(x_data[i, :, :], l_param, γ2_range, n_samples)
    end

    I_2_y = zeros(eltype(dγ[1]), length(γ1_range_LBC) - 2, length(γ2_range))
    for i in 1:length(γ2_range)
        I_2_y[:, i] = run_scheme_2_loc(x_data[:, i, :], l_param, γ1_range, n_samples)
    end

    I_2 = sqrt.((I_2_x .^ 2)[1:(end - 1), :] .+ (I_2_y .^ 2)[:, 1:(end - 1)])
    return I_2
end

function run_scheme_2(x_data, l_param, γ_range, γ_range_LBC, dγ, n_samples)
    I_2 = run_scheme_2_loc(x_data, l_param, γ_range, n_samples)

    return I_2
end

function run_scheme_2_loc(x_data, l_param, γ_range, n_samples)
    I_2_loc = zeros(eltype(γ_range[1]), length(γ_range) - 1)

    for γ_tar_indx in 1:(length(γ_range) - 1)
        range_I, range_II = get_scheme_2_ranges(γ_range, Int(l_param), γ_tar_indx)
        I_2_loc[γ_tar_indx] = run_scheme_2_fixed_bipartition(x_data,
            γ_range,
            range_I,
            range_II, n_samples)
    end

    return I_2_loc
end

function run_scheme_2_fixed_bipartition(x_data, γ_range, range_I, range_II, n_samples)
    mean_error = zero(eltype(γ_range[1]))

    for j in range_I
        for i in 1:n_samples
            mean_error += get_error_scheme_2(x_data,
                γ_range,
                range_I,
                range_II,
                get_sample([j], x_data)) / (2 * length(range_I))
        end
    end

    for j in range_II
        for i in 1:n_samples
            mean_error += get_error_scheme_2(x_data,
                γ_range,
                range_I,
                range_II,
                get_sample([j], x_data)) / (2 * length(range_II))
        end
    end

    return 1 - 2 * mean_error / n_samples
end

function get_error_scheme_2(x_data, γ_range, range_I, range_II, sampl)
    prob_1 = zero(eltype(γ_range[1]))
    prob_2 = zero(eltype(γ_range[1]))

    for i in range_I
        prob_1 += get_probability(sampl, [i], x_data) / length(range_I)
    end

    for i in range_II
        prob_2 += get_probability(sampl, [i], x_data) / length(range_II)
    end

    return minimum([
        prob_1 / (prob_1 + prob_2 + eps(eltype(γ_range[1]))),
        prob_2 / (prob_1 + prob_2 + eps(eltype(γ_range[1]))),
    ])
end
