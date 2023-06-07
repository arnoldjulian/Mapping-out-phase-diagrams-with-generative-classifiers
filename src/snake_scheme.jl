# snake
struct Snake
    vertices::Matrix{Float32}
    n_vertices::Int
    PBC::Bool
    FBC::Bool
    nodes::Array{Float32,3}
    n_nodes::Int
    widths::Vector{Float32}
    normals::Matrix{Float32}
    vertices_update::Matrix{Float32}
    α::Float32
    β::Float32
    γ::Float32
    w0::Float32
    w_end::Float32
    w_decay::Float32
    lr::Float32
    lr_decay::Float32
    energies::Vector{Float32}
    boundaries::Matrix{Float32}
    p1_range::Vector{Float32}
    p2_range::Vector{Float32}
    reshape::Bool
    train_local::Bool
    max_move::Bool
    max_moves::Vector{Float32}
    treat_boundary::Bool
end

function initialize_snake(vertices, boundaries, p1_range, p2_range;PBC=false,FBC=false,α=0.002f0,β=0.4f0,γ=0.25f0,w0=1.0f0,w_end=1.0f0,w_decay=1.0f0,lr=0.0001f0,lr_decay=0.999f0,n_nodes=4,reshape=false,train_local=false,max_move=false,max_moves=[1.0f0,1.0f0],treat_boundary=false)
    snake = Snake(vertices,size(vertices)[1],PBC,FBC,zeros(eltype(w0),size(vertices)[1],n_nodes,size(vertices)[2]),n_nodes,w0*ones(eltype(w0),size(vertices)[1]),zeros(eltype(vertices[1,1]),size(vertices)),zeros(eltype(w0),size(vertices)),α,β,γ,w0,w_end,w_decay,lr,lr_decay,[0.0f0,0.0f0,0.0f0],boundaries,p1_range,p2_range,reshape,train_local,max_move,max_moves,treat_boundary)
    set_nodes!(snake)
    return snake
end

function set_normals!(snake::Snake)
    for i in 1:size(snake.vertices)[1]
        if i == 1
            e = snake.vertices[i+1,:].-snake.vertices[i,:]
        elseif i == size(snake.vertices)[1]
            e = snake.vertices[i,:].-snake.vertices[i-1,:]
        else
            e1 = snake.vertices[i+1,:].-snake.vertices[i,:]
            e2 = snake.vertices[i,:].-snake.vertices[i-1,:]
            e = (e1.+e2)./2
        end

        snake.normals[i,:] = nullspace(e')[:,1]
    end
    return nothing
end

function set_nodes!(snake::Snake)
    set_normals!(snake)
    for i in 1:snake.n_vertices
        for j in 1:snake.n_nodes
            if j <= snake.n_nodes/2
                snake.nodes[i,j,:] = snake.vertices[i,:].+(Int((snake.n_nodes/2-j)+1)*snake.widths[i].*snake.normals[i,:])
            else
                snake.nodes[i,j,:] = snake.vertices[i,:].+((-1*(j - Int(snake.n_nodes/2))*snake.widths[i]).*snake.normals[i,:])
            end
        end
    end
    return nothing
end

function guesser(p,p_g,σ)
    S_A = -1
    x = S_A*(p-p_g)/σ
    guess = 1/(1+exp(-x))
    return guess
end

function get_loss_LBC_fast_2D_snake(data, n_samples, nD_p_space, oneD_p_space, p_g, σ,p1_range,p2_range)
    data_matrix = hcat(map(p->bilinear_interpol(data,p1_range,p2_range,nD_p_space[p,:]),1:size(nD_p_space)[1])...)
    guesses = reshape(map(p->GCPT.guesser(p,p_g,σ),oneD_p_space),(1,size(data_matrix)[2]))

    p1 = sum(data_matrix.*guesses,dims=2)[:,1]
    p2 = sum(data_matrix.*(ones(eltype(guesses[1,1]),size(guesses)).-guesses),dims=2)[:,1]
    pred_opt = [if p1[i]+p2[i] > eps(eltype(guesses[1,1])) p1[i]/(p1[i]+p2[i]) else zero(eltype(guesses[1,1])) end for i in 1:size(data_matrix)[1]]

    if count(x->x>one(eltype(guesses[1,1])), pred_opt) > 0
        error("Predictions are not normalized!")
    end
    
    return sum(map(p_indx->((crossentropy.(pred_opt, guesses[p_indx])')*(@view data_matrix[:, p_indx])),1:length(guesses)))/length(guesses)
end

function get_α_term(snake::Snake,vertices)
    α_term = zero(eltype(snake.widths[1]))
    
    # calculate the total arc-length
    arc_length = zero(eltype(snake.α))
    for i in 1:snake.n_vertices-1
        arc_length += norm(vertices[i+1,:] - vertices[i,:])
    end

    # average length for each segment
    seg_length = arc_length/(snake.n_vertices-1)

    for i in 1:snake.n_vertices
        if i == 1
            α_term += norm((vertices[2,:].-vertices[1,:])./seg_length)^2
        elseif i == snake.n_vertices
            α_term += norm((vertices[i,:].-vertices[i-1,:])./seg_length)^2
        else
            α_term += norm((vertices[i+1,:].-vertices[i-1,:])./(2*seg_length))^2
        end
    end
    return α_term
end

function get_β_term(snake::Snake,vertices)
    β_term = zero(eltype(snake.widths[1]))
    
    # calculate the total arc-length
    arc_length = zero(eltype(snake.α))
    for i in 1:snake.n_vertices-1
        arc_length += norm(vertices[i+1,:] - vertices[i,:])
    end

    # average length for each segment
    seg_length = arc_length/(snake.n_vertices-1)

    if snake.treat_boundary
        set = 1:snake.n_vertices
    else
        set = 2:snake.n_vertices-1
    end

    for i in set
        if i == 1
            β_term += norm((vertices[3,:].-(2*vertices[2,:]).+vertices[1,:])./(seg_length)^2)^2
        elseif i == snake.n_vertices
            β_term += norm((vertices[i,:].-(2*vertices[i-1,:]).+vertices[i-2,:])./(seg_length)^2)^2
        else
            β_term += norm((vertices[i+1,:].+vertices[i-1,:].-(2*vertices[i,:]))./(seg_length)^2)^2
        end
    end

    return β_term
end

function clip_snake!(snake::Snake)
    for i in 1:snake.n_vertices
        for j in 1:size(snake.vertices)[2]
            if snake.vertices[i,j] < snake.boundaries[j,1]
                snake.vertices[i,j] = snake.boundaries[j,1]
                #println("clipped vertices")
            elseif snake.vertices[i,j] > snake.boundaries[j,2]
                snake.vertices[i,j] = snake.boundaries[j,2]
                #println("clipped vertices")
            end
        end
    end
    return nothing
end

function clip_nodes(snake::Snake,node_list)
    for i in 1:snake.n_nodes
        for j in 1:size(snake.vertices)[2]
            if node_list[i,j] < snake.boundaries[j,1]
                node_list[i,j] = snake.boundaries[j,1]
                #println("clipped nodes")
            elseif node_list[i,j] > snake.boundaries[j,2]
                node_list[i,j] = snake.boundaries[j,2]
                #println("clipped nodes")
            end
        end
    end
    return node_list
end

function reshape_snake(snake::GCPT.Snake,vertices)
    # calculate the total arc-length
    arc_length = zero(eltype(snake.α))
    for i in 1:snake.n_vertices-1
        arc_length += norm(vertices[i+1,:] .- vertices[i,:])
    end

    # average length for each segment
    seg_length = arc_length/(snake.n_vertices-1)

    for i in 2:snake.n_vertices-1
        tan_direction = vertices[i,:] - vertices[i-1,:]
        tan_direction = tan_direction/norm(tan_direction)
        vertices[i,:] = vertices[i-1,:] + tan_direction * seg_length
    end
    return vertices
end

function get_grad(snake::Snake, data, n_samples)
    grad_int = zeros(eltype(snake.widths[1]),size(snake.vertices))
    grad_ext = zeros(eltype(snake.widths[1]),size(snake.vertices))

    p0 = [zero(eltype(snake.widths[1]))]
    Threads.@threads for i in 1:snake.n_vertices
        oneD_p_space = (collect(1:snake.n_nodes+1).-Int(snake.n_nodes/2+1)).*snake.widths[i]
        deleteat!(oneD_p_space,Int(snake.n_nodes/2+1))
        
        if snake.train_local
            loss = p -> get_loss_LBC_fast_2D_snake(data, n_samples, clip_nodes(snake,snake.nodes[i,:,:]), oneD_p_space, p[1], snake.widths[i],snake.p1_range,snake.p2_range)
            grad_forward = x -> ForwardDiff.gradient(loss, x)
        else
            loss = p -> get_loss_LBC_fast_2D_snake(data, n_samples, clip_nodes(snake,snake.nodes[i,:,:].-p[1].*repeat(snake.normals[i,:]',snake.n_nodes,1)), oneD_p_space.+p[1], p[1], snake.widths[i]/10,snake.p1_range,snake.p2_range)
            grad_forward = x -> FiniteDiff.finite_difference_gradient(loss, x)
        end

        grad_ext[i,:] = -1*grad_forward(p0)[1].*snake.normals[i,:]
    end
    internal_energy = p -> snake.α*get_α_term(snake,p) + snake.β*get_β_term(snake,p)
    grad_forward = x -> ForwardDiff.gradient(internal_energy, x)
    
    grad_int = grad_forward(snake.vertices)
    return grad_int, grad_ext
end

function update_snake!(snake::Snake, data, n_samples, opt_internal_losses, opt_external_losses,epoch)
    grad_int,grad_ext = get_grad(snake, data, n_samples)

    snake.vertices_update .= zeros(eltype(snake.vertices[1,1]),size(snake.vertices))
    Flux.Optimise.update!(opt_internal_losses, snake.vertices_update, grad_int)
    Flux.Optimise.update!(opt_external_losses, snake.vertices_update, grad_ext)

    if snake.FBC 
        snake.vertices_update[1,:] = zeros(eltype(snake.vertices_update[1,1]),size(snake.vertices_update[1,:]))
        snake.vertices_update[snake.n_vertices,:] = zeros(eltype(snake.vertices_update[1,1]),size(snake.vertices_update[1,:]))
    end

    if snake.max_move
        snake.vertices[:,1] .+= snake.max_moves[1].*(tanh.(snake.vertices_update[:,1].-snake.vertices[:,1]))
        snake.vertices[:,2] .+= snake.max_moves[2].*(tanh.(snake.vertices_update[:,2].-snake.vertices[:,2]))
    else
        snake.vertices .+= snake.vertices_update
    end

    # reshape snake
    if snake.reshape
        frwrd = reshape_snake(snake,deepcopy(snake.vertices))
        bckwrd = reverse(reshape_snake(snake,reverse(deepcopy(snake.vertices),dims=1)),dims=1)
        snake.vertices .= (frwrd .+ bckwrd)./ 2
    end

    clip_snake!(snake)
    snake.widths .= width_sched(snake,epoch)*ones(eltype(snake.widths[1]),size(snake.widths))
    set_nodes!(snake)
    
    return nothing
end

function width_sched(snake,epoch) 
    return snake.w_end+(snake.w0-snake.w_end)*(snake.w_decay^epoch)
end 

function pnt_neighbours(p_range,pnt)
    arg = argmin(abs.(p_range.-pnt))
    c = p_range[arg]

    if arg == length(p_range) 
        c_1 = p_range[length(p_range)-1]
        arg_1 = length(p_range)-1
        c_2 = c
        arg_2 = arg
    elseif arg == 1
        c_1 = c
        arg_1 = arg
        c_2 = p_range[2]
        arg_2 = 2
    elseif abs(pnt) < abs(c)
        c_1 = p_range[arg-1]
        arg_1 = arg-1
        c_2 = c
        arg_2 = arg
    else #abs(pnt) > abs(c)
        c_1 = c
        arg_1 = arg
        c_2 = p_range[arg+1]
        arg_2 = arg+1
    end
    return c_1,c_2,arg_1,arg_2
end

function bilinear_interpol(data,p1_range,p2_range,pnt)
    x_1,x_2,arg_x_1,arg_x_2 = pnt_neighbours(p1_range,pnt[1])
    y_1,y_2,arg_y_1,arg_y_2 = pnt_neighbours(p2_range,pnt[2])

    dxy = (x_2 - x_1)*(y_2 - y_1)

    a = x_2 - pnt[1]
    b = pnt[1] - x_1
    c = y_2 - pnt[2]
    d = pnt[2] - y_1

    w11 = a*c/dxy
    w12 = a*d/dxy
    w21 = b*c/dxy
    w22 = b*d/dxy
    
    return w11.*(@view data[arg_x_1,arg_y_1,:]) .+ w12.*(@view data[arg_x_1,arg_y_2,:]) .+ w21.*(@view data[arg_x_2,arg_y_1,:]) .+ w22.*(@view data[arg_x_2,arg_y_2,:])
end

function rescale(p,p_range)
    p_ren = (p-p_range[1])/maximum(abs.(p_range.-p_range[1]))
    return p_ren
end

function inv_rescale(p,p_range)
    p_ren = p*maximum(abs.(p_range.-p_range[1]))+p_range[1]
    return p_ren
end