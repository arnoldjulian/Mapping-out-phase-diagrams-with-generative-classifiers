#######################################################
# scheme 1 with expected values being evaluated exactly
#######################################################
function run_scheme_1(x_data, class_data_prov, dp)
    param_space_dim = length(size(x_data)) - 1
    class_data = get_new_class_data(x_data, class_data_prov, param_space_dim)
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
        I_1_classes = zeros(eltype(dp[1]), size(x_data)[1] - 2, n_classes)

        for i in 1:n_classes
            grad = (circshift(@view(mean_pred[:, i]), (-1)) .-
                    circshift(@view(mean_pred[:, i]), (1))) ./ (2 * dp[1])
            I_1_classes[:, i] = @view sqrt.(grad .^ 2)[2:(end - 1)]
        end

    elseif param_space_dim == 2
        I_1_classes = zeros(eltype(dp[1]),
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

    return mean_pred, I_1_classes, I_1
end

function get_new_class_data(x_data, class_data, param_space_dim)
    n_classes = length(class_data)

    if param_space_dim == 1
        new_class_data = zeros(Int64, size(x_data)[1], n_classes)
        for class in 1:n_classes
            for indx in 1:length(class_data[class])
                new_class_data[class_data[class][indx][1], class] = 1
            end
        end
    elseif param_space_dim == 2
        new_class_data = zeros(Int64, size(x_data)[1], size(x_data)[2], n_classes)
        for class in 1:n_classes
            for indx in 1:length(class_data[class])
                new_class_data[class_data[class][indx][1], class_data[class][indx][2], class] = 1
            end
        end
    else
        error("Parameter spaces with dimension > 2 are currently not supported.")
    end

    return new_class_data
end

############################################################################################################################################################
# scheme 1 with expectation values being approximated with sample mean, where n_samples denotes the number of samples drawn at each point in parameter space
############################################################################################################################################################

# using generative models given in the form of x_data object
function run_scheme_1(x_data, class_data, dγ, n_samples)
    param_space_dim = length(size(x_data)) - 1
    n_classes = size(class_data)[end]

    if param_space_dim == 1
        mean_pred = zeros(eltype(dγ[1]), (size(x_data)[1], 1, length(class_data)))
        for j in 1:size(x_data)[1]
            for i in 1:n_samples
                mean_pred[j, 1, :] .+= get_pred_scheme_1(x_data,
                    class_data,
                    dγ,
                    get_sample([j], x_data))
            end
        end
        mean_pred .= mean_pred ./ n_samples

        I_1_classes = zeros(eltype(dγ[1]), size(x_data)[1] - 2, n_classes)

        for i in 1:n_classes
            grad = (circshift(@view(mean_pred[:, :, i]), (-1)) .-
                    circshift(@view(mean_pred[:, :, i]), (1))) ./ (2 * dγ[1])

            I_1_classes[:, i] = @view sqrt.(grad .^ 2)[2:(end - 1)]
        end

    elseif param_space_dim == 2
        mean_pred = zeros(eltype(dγ[1]),
            (size(x_data)[1], size(x_data)[2], length(class_data)))
        for k in 1:size(x_data)[2]
            for j in 1:size(x_data)[1]
                for i in 1:n_samples
                    mean_pred[j, k, :] .+= get_pred_scheme_1(x_data,
                        class_data,
                        dγ,
                        get_sample([j, k], x_data))
                end
            end
        end
        mean_pred .= mean_pred ./ n_samples

        I_1_classes = zeros(eltype(dγ[1]),
            size(x_data)[1] - 2,
            size(x_data)[2] - 2,
            n_classes)

        for i in 1:n_classes
            grad_1 = (circshift(@view(mean_pred[:, :, i]), (-1, 0)) .-
                      circshift(@view(mean_pred[:, :, i]), (1, 0))) ./ (2 * dγ[1])
            grad_2 = (circshift(@view(mean_pred[:, :, i]), (0, -1)) .-
                      circshift(@view(mean_pred[:, :, i]), (0, 1))) ./ (2 * dγ[2])
            I_1_classes[:, :, i] = @view sqrt.(grad_1 .^ 2 .+ grad_2 .^ 2)[2:(end - 1),
                2:(end - 1)]
        end

    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    I_1 = sum(I_1_classes, dims = param_space_dim + 1) ./ n_classes

    return mean_pred, I_1_classes, I_1
end

function get_pred_scheme_1(x_data, class_data, dγ, sampl)
    probs = zeros(eltype(dγ[1]), length(class_data))

    if length(dγ) == 1
        for i in 1:length(class_data)
            for γ in class_data[i]
                probs[i] += get_probability(sampl, [γ[1]], x_data)
            end
        end
    else
        for i in 1:length(class_data)
            for γ in class_data[i]
                probs[i] += get_probability(sampl, [γ[1], γ[2]], x_data)
            end
        end
    end

    if sum(probs) > eps(eltype(dγ[1]))
        return probs ./ sum(probs)
    else
        return zeros(eltype(dγ[1]), length(class_data))
    end
end

# using generative models given in the form of exact wavefunctions
function run_scheme_1_wf(wavefunc_data,
    class_data,
    dγ,
    n_samples,
    L;
    savitzky = true,
    sav_wind = 21,
    sav_order = 4)
    param_space_dim = length(dγ)
    n_classes = size(class_data)[end]

    if param_space_dim == 1
        sampl = ([0], zeros(eltype(wavefunc_data[1, 1]), (2^L, 2^L)))
        mean_pred = zeros(eltype(dγ[1]), (size(wavefunc_data)[1], 1, length(class_data)))
        for j in 1:size(wavefunc_data)[1]
            for i in 1:n_samples
                mean_pred[j, 1, :] .+= get_pred_scheme_1_wf(wavefunc_data,
                    class_data,
                    dγ,
                    get_sample_wf!([j], wavefunc_data, L, sampl), L)
            end
        end
        mean_pred .= mean_pred ./ n_samples

        I_1_classes = zeros(eltype(dγ[1]), size(wavefunc_data)[1] - 2, n_classes)

        for i in 1:n_classes
            if savitzky
                grad = savitzky_golay(@view(mean_pred[:, 1, i]),
                    sav_wind,
                    sav_order,
                    deriv = 1,
                    rate = 1 / dγ[1]).y
            else
                grad = (circshift(@view(mean_pred[:, :, i]), (-1)) .-
                        circshift(@view(mean_pred[:, :, i]), (1))) ./ (2 * dγ[1])
            end

            I_1_classes[:, i] = @view sqrt.(grad .^ 2)[2:(end - 1)]
        end

    elseif param_space_dim == 2
        sampl = ([0], zeros(eltype(wavefunc_data[1, 1, 1]), (2^L, 2^L)))
        mean_pred = zeros(eltype(dγ[1]),
            (size(wavefunc_data)[1], size(wavefunc_data)[2], length(class_data)))
        for k in 1:size(wavefunc_data)[2]
            for j in 1:size(wavefunc_data)[1]
                for i in 1:n_samples
                    mean_pred[j, k, :] .+= get_pred_scheme_1_wf(wavefunc_data,
                        class_data,
                        dγ,
                        get_sample_wf!([j, k], wavefunc_data, L, sampl), L)
                end
            end
        end
        mean_pred .= mean_pred ./ n_samples

        I_1_classes = zeros(eltype(dγ[1]),
            size(x_data)[1] - 2,
            size(x_data)[2] - 2,
            n_classes)

        for i in 1:n_classes
            grad_1 = (circshift(@view(mean_pred[:, :, i]), (-1, 0)) .-
                      circshift(@view(mean_pred[:, :, i]), (1, 0))) ./ (2 * dγ[1])
            grad_2 = (circshift(@view(mean_pred[:, :, i]), (0, -1)) .-
                      circshift(@view(mean_pred[:, :, i]), (0, 1))) ./ (2 * dγ[2])
            I_1_classes[:, :, i] = @view sqrt.(grad_1 .^ 2 .+ grad_2 .^ 2)[2:(end - 1),
                2:(end - 1)]
        end

    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    I_1 = sum(I_1_classes, dims = param_space_dim + 1) ./ n_classes

    return mean_pred, I_1_classes, I_1
end

function get_pred_scheme_1_wf(wavefunc_data, class_data, dγ, sampl, L)
    probs = zeros(eltype(dγ[1]), length(class_data))

    if length(dγ) == 1
        for i in 1:length(class_data)
            for γ in class_data[i]
                probs[i] += get_probability_wf(sampl, [γ[1]], wavefunc_data, L)
            end
        end
    else
        for i in 1:length(class_data)
            for γ in class_data[i]
                probs[i] += get_probability_wf(sampl, [γ[1], γ[2]], wavefunc_data, L)
            end
        end
    end

    if sum(probs) > eps(eltype(dγ[1]))
        return probs ./ sum(probs)
    else
        return zeros(eltype(dγ[1]), length(class_data))
    end
end

# using generative models given in the form of MPS wavefunctions
function run_scheme_1_MPS(MPS_data,
    class_data,
    dγ,
    n_samples,
    L,
    sites;
    savitzky = true,
    sav_wind = 21,
    sav_order = 4)
    param_space_dim = length(dγ)
    n_classes = size(class_data)[end]

    if param_space_dim == 1
        mean_pred = zeros(eltype(dγ[1]), (size(MPS_data)[1], 1, length(class_data)))
        for j in 1:size(MPS_data)[1]
            for i in 1:n_samples
                mean_pred[j, 1, :] .+= get_pred_scheme_1_MPS(MPS_data,
                    class_data,
                    dγ,
                    get_sample_MPS([j], MPS_data, L, sites), L, sites)
            end
        end
        mean_pred .= mean_pred ./ n_samples

        I_1_classes = zeros(eltype(dγ[1]), size(MPS_data)[1] - 2, n_classes)

        for i in 1:n_classes
            if savitzky
                grad = savitzky_golay(@view(mean_pred[:, 1, i]),
                    sav_wind,
                    sav_order,
                    deriv = 1,
                    rate = 1 / dγ[1]).y
            else
                grad = (circshift(@view(mean_pred[:, :, i]), (-1)) .-
                        circshift(@view(mean_pred[:, :, i]), (1))) ./ (2 * dγ[1])
            end

            I_1_classes[:, i] = @view sqrt.(grad .^ 2)[2:(end - 1)]
        end

    elseif param_space_dim == 2
        mean_pred = zeros(eltype(dγ[1]),
            (size(MPS_data)[1], size(MPS_data)[2], length(class_data)))
        for k in 1:size(MPS_data)[2]
            for j in 1:size(MPS_data)[1]
                for i in 1:n_samples
                    mean_pred[j, k, :] .+= get_pred_scheme_1_MPS(MPS_data,
                        class_data,
                        dγ,
                        get_sample_MPS([j, k], MPS_data, L, sites), L, sites)
                end
            end
        end
        mean_pred .= mean_pred ./ n_samples

        I_1_classes = zeros(eltype(dγ[1]),
            size(x_data)[1] - 2,
            size(x_data)[2] - 2,
            n_classes)

        for i in 1:n_classes
            grad_1 = (circshift(@view(mean_pred[:, :, i]), (-1, 0)) .-
                      circshift(@view(mean_pred[:, :, i]), (1, 0))) ./ (2 * dγ[1])
            grad_2 = (circshift(@view(mean_pred[:, :, i]), (0, -1)) .-
                      circshift(@view(mean_pred[:, :, i]), (0, 1))) ./ (2 * dγ[2])
            I_1_classes[:, :, i] = @view sqrt.(grad_1 .^ 2 .+ grad_2 .^ 2)[2:(end - 1),
                2:(end - 1)]
        end

    else
        error("Parameter spaces with dimension > 2 are currently not supported. Need to implement the corresponding derivative.")
    end

    I_1 = sum(I_1_classes, dims = param_space_dim + 1) ./ n_classes

    return mean_pred, I_1_classes, I_1
end

function get_pred_scheme_1_MPS(MPS_data, class_data, dγ, sampl, L, sites)
    probs = zeros(eltype(dγ[1]), length(class_data))

    if length(dγ) == 1
        for i in 1:length(class_data)
            for γ in class_data[i]
                probs[i] += get_probability_MPS(sampl, [γ[1]], MPS_data, L, sites)
            end
        end
    else
        for i in 1:length(class_data)
            for γ in class_data[i]
                probs[i] += get_probability_MPS(sampl, [γ[1], γ[2]], MPS_data, L, sites)
            end
        end
    end

    if sum(probs) > eps(eltype(dγ[1]))
        return probs ./ sum(probs)
    else
        return zeros(eltype(dγ[1]), length(class_data))
    end
end

# generative models of other forms can be used by overloading the `get_probability` and `get_sample` functions in utils.jl
