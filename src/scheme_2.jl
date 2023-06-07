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

# compute optimal indicator and loss of LBC across entire range of tuning parameter
function run_scheme_2_loc(data, p_range, l_param)
    I_2_loc = zeros(eltype(p_range[1]), length(p_range) - 1)

    # start parallel computation for sampled values of tuning parameter
    for p_tar_indx in 1:(length(p_range) - 1)
        range_I, range_II = get_scheme_2_ranges(p_range, Int(l_param / 2), p_tar_indx)
        I_2_loc[p_tar_indx] = run_scheme_2_fixed_bipartition(data,
            p_range,
            range_I,
            range_II)
    end

    return I_2_loc
end

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
