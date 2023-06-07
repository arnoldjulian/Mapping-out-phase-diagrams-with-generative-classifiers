function run_scheme_3(x_data, p_data, dp)
    param_space_dim = length(size(x_data)) - 1
    n_samples = size(x_data)[end]
    n_param_space_points = Int(length(x_data) / n_samples)

    x_data_r = reshape(x_data, (n_param_space_points, n_samples))'
    p_data_r = reshape(p_data, (n_param_space_points, param_space_dim))
    pred = zeros(eltype(dp[1]), n_samples, param_space_dim)
    mean_pred_r = zeros(eltype(dp[1]), n_param_space_points, param_space_dim)
    mean_pred_sq_r = zeros(eltype(dp[1]), n_param_space_points, param_space_dim)

    for i in 1:param_space_dim
        pred[:, i] = sum(x_data_r .* (@view p_data_r[:, i])', dims = 2) ./
                     (sum(x_data_r, dims = 2) .+ eps(eltype(dp[i])))
        mean_pred_r[:, i] = sum(x_data_r .* pred[:, i], dims = 1)[1, :]
        mean_pred_sq_r[:, i] = sum(x_data_r .* (pred[:, i] .^ 2), dims = 1)[1, :]
    end
    mean_pred = reshape(mean_pred_r, size(p_data))
    mean_pred_sq = reshape(mean_pred_sq_r, size(p_data))

    std = sqrt.(mean_pred_sq .- (mean_pred .^ 2))

    if param_space_dim == 1
        I_3 = (((circshift((@view mean_pred[:, 1]), (-1)) .- circshift((@view mean_pred[:, 1]), (1))) ./ (2 * dp[1]))[2:(end - 1)]) ./
              std[2:(end - 1)]
    elseif param_space_dim == 2
        sig_x = ((circshift((@view mean_pred[:, :, 1]), (-1, 0)) .- circshift((@view mean_pred[:, :, 1]), (1, 0))) ./ (2 * dp[1]))[2:(end - 1),
            2:(end - 1)]
        sig_y = ((circshift((@view mean_pred[:, :, 2]), (0, -1)) .- circshift((@view mean_pred[:, :, 2]), (0, 1))) ./ (2 * dp[2]))[2:(end - 1),
            2:(end - 1)]
        I_3 = sqrt.(((sig_x ./ std[2:(end - 1), 2:(end - 1), 1]) .^ 2) .+
                    ((sig_y ./ std[2:(end - 1), 2:(end - 1), 2]) .^ 2))
    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    return mean_pred, std, I_3
end

function run_scheme_3_linescans(x_data, p_data, dγ)
    γ1_range = p_data[:, 1, 1]
    γ2_range = p_data[1, :, 2]

    I_3_x = zeros(eltype(Float32), length(γ1_range) - 2, length(γ2_range))
    I_3_y = zeros(eltype(Float32), length(γ1_range), length(γ2_range) - 2)

    for i in 1:length(γ1_range)
        _, _, I_3_y_shot = run_scheme_3(x_data[i, :, :], p_data[i, :, 2], [dγ[2]])
        I_3_y[i, :] .= I_3_y_shot
    end

    for i in 1:length(γ2_range)
        _, _, I_3_x_shot = run_scheme_3(x_data[:, i, :], p_data[:, i, 1], [dγ[1]])
        I_3_x[:, i] .= I_3_x_shot
    end

    I_3 = sqrt.((I_3_x[:, 2:(end - 1)] .^ 2) .+ (I_3_y[2:(end - 1), :] .^ 2))
    return I_3
end
