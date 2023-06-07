function run_scheme_1(x_data, class_data, dp)
    param_space_dim = length(size(x_data)) - 1
    n_samples = size(x_data)[end]
    n_param_space_points = Int(length(x_data) / n_samples)
    n_classes = size(class_data)[end]

    x_data_r = reshape(x_data, (n_param_space_points, n_samples))'
    class_data_r = reshape(class_data, (n_param_space_points, n_classes))
    pred_x_prov = zeros(eltype(x_data_r[1, 1]), size(x_data_r)[1], n_classes)
    pred_x = zeros(eltype(x_data_r[1, 1]), size(x_data_r)[1], n_classes)
    mean_pred_r = zeros(eltype(x_data_r[1, 1]), n_param_space_points, n_classes)

    pred_x_prov_sum = sum(x_data_r * class_data_r, dims = 2)
    for i in 1:(n_classes - 1)
        pred_x_prov[:, i] = sum(x_data_r .* (@view class_data_r[:, i])', dims = 2)
        pred_x[:, i] = [if pred_x_prov_sum[j] != zero(eltype(x_data_r[1, 1]))
            pred_x_prov[j, i] / (pred_x_prov_sum[j])
        else
            zero(eltype(x_data_r[1, 1]))
        end
                        for j in 1:size(x_data_r)[1]]
    end
    pred_x[:, n_classes] = ones(eltype(x_data_r[1, 1]), size(x_data_r)[1]) .-
                           sum(pred_x, dims = 2)

    for i in 1:n_classes
        mean_pred_r[:, i] = sum(x_data_r .* pred_x[:, i], dims = 1)[1, :]
    end
    mean_pred = reshape(mean_pred_r, size(class_data))

    if param_space_dim == 1
        I_1_classes = zeros(eltype(x_data_r[1, 1]), size(x_data)[1] - 2, n_classes)

        for i in 1:n_classes
            grad = (circshift(@view(mean_pred[:, i]), (-1)) .-
                    circshift(@view(mean_pred[:, i]), (1))) ./ (2 * dp[1])
            I_1_classes[:, i] = @view sqrt.(grad .^ 2)[2:(end - 1)]
        end

    elseif param_space_dim == 2
        I_1_classes = zeros(eltype(x_data_r[1, 1]),
            size(x_data)[1] - 2,
            size(x_data)[2] - 2,
            n_classes)

        for i in 1:n_classes
            grad_1 = (circshift(@view(mean_pred[:, :, i]), (-1, 0)) .-
                      circshift(@view(mean_pred[:, :, i]), (1, 0))) ./ (2 * dp[1])
            grad_2 = (circshift(@view(mean_pred[:, :, i]), (0, -1)) .-
                      circshift(@view(mean_pred[:, :, i]), (0, 1))) ./ (2 * dp[2])
            I_1_classes[:, :, i] = @view sqrt.(grad_1 .^ 2 .+ grad_2 .^ 2)[2:(end - 1),
                2:(end - 1)]
        end
    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    I_1 = sum(I_1_classes, dims = param_space_dim + 1) ./ n_classes

    return I_1
    #return mean_pred, I_1_classes, I_1
end
