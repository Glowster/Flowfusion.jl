#Note: Haven't figured out exactly what, in the literature, this is. Not very tested!

struct DoobMatchingFlow{Proc, B, F} <: Process
    P::Proc
    onescale::B #Controls whether the "step" is unit scale or "time remaining" scale. Need to think carefully about schedules in all this...
    transform::F #Transforms the output of the model to the rate space. Must act on the whole tensor.
    #Note: losses can be compared for different transforms, but not for different onescale.
end

DoobMatchingFlow(P::DiscreteProcess) = DoobMatchingFlow(P, true, NNlib.softplus) #x -> exp.(clamp.(x, -100, 11)) also works, but is scary
DoobMatchingFlow(P::DiscreteProcess, transform::Function) = DoobMatchingFlow(P, true, transform)
DoobMatchingFlow(P::DiscreteProcess, onescale::Bool) = DoobMatchingFlow(P, onescale, NNlib.softplus)

onescale(P::DoobMatchingFlow,t) = P.onescale ? (1 .- t)  : eltype(t)(1)
mulexpand(t,x) = expand(t, ndims(x)) .* x

Flowfusion.bridge(p::DoobMatchingFlow, x0::DiscreteState{<:AbstractArray{<:Signed}}, x1::DiscreteState{<:AbstractArray{<:Signed}}, t) = bridge(p.P, x0, x1, t)

function fallback_doob(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState; delta = eltype(t)(1e-5))
    return (tensor(forward(Xt, P, delta) ⊙ backward(X1, P, (1 .- t) .- delta)) .- tensor(onehot(Xt))) ./ delta;
end

doob_guide(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState) = fallback_doob(P, t, Xt, X1)

function closed_form_doob(P::DiscreteProcess, t, Xt::DiscreteState, X1::DiscreteState)
    tenXt = tensor(onehot(Xt))
    bk = tensor(backward(X1, P, 1 .- t))
    fv = forward_positive_velocities(onehot(Xt), P)
    positive_doob = (fv .* bk) ./ sum(bk .* tenXt, dims = 1)
    return positive_doob .- tenXt .* sum(positive_doob, dims = 1)
end

forward_positive_velocities(Xt::DiscreteState, P::PiQ)= (P.r .* (P.π ./ sum(P.π))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::PiQ, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)
forward_positive_velocities(Xt::DiscreteState, P::UniformUnmasking{T}) where T = (P.μ .* T((1 ./ (Xt.K-1)))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::UniformUnmasking, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)
forward_positive_velocities(Xt::DiscreteState, P::UniformDiscrete{T}) where T = (P.μ * T(1/(Xt.K*(1-1/Xt.K)))) .* (1 .- tensor(onehot(Xt)))
doob_guide(P::UniformDiscrete, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)

function old_forward_positive_velocities(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt = onehot(Xt) #I believe this does not modify Xt if it is already onehot
    Q = zeros(Float64, size(Xt.state))
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)
    # display(size(Q))
    # display(size(batch_indices)) 
    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        length(idx) <= 1 && continue
        u = node.u
        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue
        for I in CartesianIndices(batch_indices)
            for j_global in idx
                if batch_indices[I] != j_global && batch_indices[I] in idx
                    Q[j_global, I[1], I[2]] += u * (π[j_global] / sum_π)
                end
            end
        end
    end
    return Q
end

function forward_positive_velocities(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt = onehot(Xt)
    #Q = zeros(Float64, size(Xt.state)) # fix type
    Q = fill!(similar(π, T, size(Xt.state)...), 0)
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)
    
    batch_dims = size(batch_indices)

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue

        idx_to_local_map = Dict(j_global => i for (i, j_global) in enumerate(idx))
        node_updates = (node.u / sum_π) .* π_partition_view

        for (I, b_idx) in pairs(batch_indices)

            local_idx = get(idx_to_local_map, b_idx, 0) # Returns 0 if not found
            if local_idx > 0
                Q_view = view(Q, idx, I)
                Q_view .+= node_updates
                Q[b_idx, I] -= node_updates[local_idx]
            end
        end
    end
    return Q
end

function forward_positive_velocities_superduperhyperfun(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt_state = onehot(Xt).state

    Q = fill!(similar(π, T, size(Xt_state)...), 0)

    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)

    batch_indices = onecold(Xt_state)

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition = π[idx]
        sum_π = sum(π_partition)
        isapprox(sum_π, zero(T)) && continue

        node_updates = (node.u / sum_π) .* π_partition

        # Create and transfer the reverse map
        reverse_map = zeros(Int, N)
        reverse_map[idx] .= 1:k
        d_reverse_map = similar(π, Int, N)
        copyto!(d_reverse_map, reverse_map)

        # Create the mask
        mask = (d_reverse_map[batch_indices]) .> 0
        !any(mask) && continue

        # Apply the main update
        Q[idx, mask] .+= node_updates

        # --- Correction Step ---

        # Get row indices (dimension 1)
        rows_to_correct = batch_indices[mask]
        
        # `findall` on the N-1 dimensional mask gives CartesianIndex{N-1}
        batch_cartesian_indices = findall(mask)

        # Extract column indices (dimension 2)
        cols_to_correct = map(ci -> ci[1], batch_cartesian_indices)
        # Extract slice indices (dimension 3)
        slices_to_correct = map(ci -> ci[2], batch_cartesian_indices)

        # --- THE FIX: Use the correct 3D formula for linear indexing ---
        D1, D2 = size(Q, 1), size(Q, 2)
        linear_indices_for_correction = (slices_to_correct .- 1) .* D1 .* D2 .+ 
                                        (cols_to_correct .- 1) .* D1 .+ 
                                        rows_to_correct

        # Get values to subtract
        local_indices_for_correction = d_reverse_map[rows_to_correct]
        values_to_subtract = node_updates[local_indices_for_correction]
        
        # Apply corrections
        Q[linear_indices_for_correction] .-= values_to_subtract
    end
    return Q
end

function forward_positive_velocities_par(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt = onehot(Xt)
    Q = fill!(similar(π, T, size(Xt.state)...), 0)
    #Q = zeros(Float64, size(Xt.state)) # fix type
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)
    batch_dims = size(batch_indices)
    
    expand_to_data_dims(v) = reshape(v, (length(v), ntuple(_ -> 1, length(batch_dims))...))
    
    expand_to_state_dim(a::AbstractArray) = reshape(a, (1, size(a)...))
    expand_to_state_dim(a) = a

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue



        node_updates = (node.u / sum_π) .* π_partition_view
        # --- This part is the same as before ---
        # Create a mask of 1s and 0s

        potential_indices = searchsortedfirst.(Ref(idx), batch_indices)
        mask = (potential_indices .<= length(idx)) .& (idx[potential_indices] .== batch_indices)
        local_indices = ifelse.(mask, potential_indices, 1)

        # idx_to_local_map = Dict(j_global => i for (i, j_global) in enumerate(idx))
        # mask = haskey.(Ref(idx_to_local_map), batch_indices) #fix this line
        # local_indices = get.(Ref(idx_to_local_map), batch_indices, 1)

        # Vectorized update for all nodes except the state itself
        Q[idx, CartesianIndices(batch_dims)] .+= expand_to_data_dims(node_updates) .* expand_to_state_dim(mask)
       
        # 2. Calculate the specific correction value for each batch element
        # println(typeof(local_indices))
        # println(local_indices)
        # println(node_updates)
        correction_values = node_updates[local_indices]

        # 3. Find the linear indices in Q that correspond to Q[b_idx, I]
        #    This is the key scatter-indexing step.

        full_3D_coords = CartesianIndex.(vec(batch_indices), vec(CartesianIndices(batch_indices)))
        dest_indices = LinearIndices(Q)[full_3D_coords]

        # println(dest_indices)
        # println(size(mask))
        # println(size(correction_values))
        # println(size(correction_values .* mask))
        # println(size(Q[dest_indices]))
        # 4. Apply the corrections, multiplied by the mask to zero out non-updates
        Q[dest_indices] .-= vec(correction_values .* mask)


       
    end
    return Q

end

function forward_positive_velocities_ok(Xt::DiscreteState, P::HPiQ{T}) where T
    tree = P.tree
    π = P.π
    
    Xt = onehot(Xt)
    # --- Initial Shape Debugging ---
    # println("--- Debugging Shapes ---")
    # println("Shape of input Xt.state: ", size(Xt.state))
    

    Q = fill!(similar(π, T, size(Xt.state)...), 0)
    # println("Shape of output matrix Q after initialization: ", size(Q))

    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    
    batch_indices = onecold(Xt.state) 
    batch_dims = size(batch_indices)
    # println("Shape of batch_indices (and batch_dims): ", batch_dims)
    # println("------------------------\n")
    
    debug_prints_done = false

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue

        node_updates = (node.u / sum_π) .* π_partition_view
        
        potential_indices = searchsortedfirst.(Ref(idx), batch_indices)
        
        mask_in_bounds = (potential_indices .<= k)
        safe_indices = ifelse.(mask_in_bounds, potential_indices, 1)
        found_values = view(idx, safe_indices)
        mask = mask_in_bounds .& (found_values .== batch_indices)

        #mask = (potential_indices .<= k) .& (view(idx, potential_indices) .== batch_indices)
        local_indices = ifelse.(mask, potential_indices, 1)

        Q_view_pos = view(Q, idx, CartesianIndices(batch_dims))
        
        update_term = reshape(node_updates, (k, ntuple(_->1, length(batch_dims))...)) .* reshape(mask, (1, batch_dims...))
        Q_view_pos .+= update_term
       
        correction_values = view(node_updates, local_indices)
        
        correction_matrix = (reshape(idx, (k, ntuple(_->1, length(batch_dims))...)) .== reshape(batch_indices, (1, batch_dims...))) .* reshape(correction_values, (1, batch_dims...))
        Q_view_pos .-= correction_matrix

        # --- Inner Loop Debugging (runs only once) ---
        # if !debug_prints_done
        #     println("--- Debugging Inside First Loop Iteration ---")
        #     println("Partition size `k`: ", k)
        #     println("Shape of `mask`: ", size(mask))
        #     println("Shape of `Q_view_pos`: ", size(Q_view_pos))
        #     println("Shape of `update_term`: ", size(update_term))
        #     println("Shape of `correction_matrix`: ", size(correction_matrix))
        #     println("-----------------------------------------\n")
        #     debug_prints_done = true
        # end
    end
    
    return Q
end

function forward_positive_velocities_par2(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    
    Q = similar(Xt.state, Float32)
    fill!(Q, 0)
    
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)

    expand_to_data_dims(v) = reshape(v, (length(v), ntuple(_ -> 1, ndims(batch_indices))...))
    expand_to_state_dim(a::AbstractArray) = reshape(a, (1, size(a)...))
    expand_to_state_dim(a) = a

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue

        node_updates_cpu = (node.u / sum_π) .* π_partition_view
        lookup_array_cpu = zeros(Int, N)
        for (i, j_global) in enumerate(idx)
            lookup_array_cpu[j_global] = i
        end

        lookup_array_gpu = adapt(Xt.state, lookup_array_cpu)
        node_updates_gpu = adapt(Xt.state, node_updates_cpu)

        raw_local_indices = lookup_array_gpu[batch_indices]
        mask = raw_local_indices .> 0
        
        Q[idx, :, :] .+= expand_to_data_dims(node_updates_gpu) .* expand_to_state_dim(mask)

        local_indices = ifelse.(mask, raw_local_indices, 1)
        correction_values = node_updates_gpu[local_indices]
        
        full_3D_coords = CartesianIndex.(vec(batch_indices), vec(CartesianIndices(batch_indices)))
        dest_indices = LinearIndices(Q)[full_3D_coords]

        Q[dest_indices] .-= vec(correction_values .* mask)
    end
    
    return Q
end

function forward_positive_velocities_par3(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt = onehot(Xt)
    Q = zeros(Float64, size(Xt.state))  # Move Q to GPU
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)  # Move indices to GPU
    batch_dims = size(batch_indices)
    
    # GPU-compatible helper functions
    expand_to_data_dims(v) = reshape(v, (length(v), ntuple(_ -> 1, length(batch_dims))...))
    expand_to_state_dim(a) = reshape(a, (1, size(a)...))

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue

        # GPU-optimized replacements
        gpu_idx = idx  # Move current indices to GPU
        
        # 1. Create mask using GPU-accelerated membership check
        mask = map(batch_indices) do x
            any(==(x), gpu_idx)
        end
        
        # 2. Create local indices mapping
        local_indices = map(batch_indices) do x
            idx_val = 1  # Default value
            for (li, global_idx) in enumerate(gpu_idx)
                if global_idx == x
                    idx_val = li
                    break
                end
            end
            return idx_val
        end

        # Vectorized update
        node_updates = (node.u / sum_π) .* π_partition_view
        Q_update = expand_to_data_dims(node_updates) .* expand_to_state_dim(mask)
        Q_gathered = view(Q, idx, :)  # Efficient view for scattered indices
        Q_gathered .+= Q_update

        # Scatter correction updates
        correction_values = node_updates[local_indices]
        linear_indices = [i + (j-1)*size(Q,1) for j in 1:prod(batch_dims) for i in idx]
        Q[linear_indices] .-= vec(correction_values .* mask)
    end
    return Q
end
function forward_positive_velocities_par4(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt = onehot(Xt)
    Q = zeros(Float64, size(Xt.state))  # Ideally, this should be a GPU array if Xt.state is.
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)
    batch_dims = size(batch_indices)

    # GPU-compatible helper functions
    expand_to_data_dims(v) = reshape(v, (length(v), ntuple(_ -> 1, length(batch_dims))...))
    expand_to_state_dim(a) = reshape(a, (1, size(a)...))

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices # Assuming idx is already on the correct device (GPU)
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue

        # 1. Create a mask for batch elements belonging to the current partition
        # This can be slow if done iteratively. A broadcasted approach is better if possible.
        mask = any(Fix2(==, batch_indices), idx)

        # Ensure the mask can be broadcasted correctly
        broadcast_mask = expand_to_state_dim(mask)

        # 2. Calculate the update rates for each state in the partition
        node_updates = (node.u / sum_π) .* π_partition_view
        
        # 3. Create a view into the relevant rows of Q
        Q_gathered = view(Q, idx, :, :)

        # 4. Perform the update using the (Sum of All) - (Self) pattern
        
        # Add the sum of all updates to every state in the partition
        total_update = sum(node_updates)
        Q_gathered .+= total_update .* broadcast_mask
        
        # Subtract the individual ("self") update from each corresponding state
        individual_updates = expand_to_data_dims(node_updates)
        Q_gathered .-= individual_updates .* broadcast_mask
    end
    return Q
end


doob_guide(P::HPiQ, t, Xt::DiscreteState, X1::DiscreteState) = closed_form_doob(P, t, Xt, X1)

Guide(P::DoobMatchingFlow, t, Xt::DiscreteState, X1::DiscreteState) = Flowfusion.Guide(mulexpand(onescale(P, t), doob_guide(P.P, t, Xt, X1)))
Guide(P::DoobMatchingFlow, t, mXt::Union{MaskedState{<:DiscreteState}, DiscreteState}, mX1::MaskedState{<:DiscreteState}) = Guide(mulexpand(onescale(P, t), doob_guide(P.P, t, mXt, mX1)), mX1.cmask, mX1.lmask)

function rate_constraint(Xt, X̂₁, f) 
    posQt = f(X̂₁) .* (1 .- Xt)   
    diagQt = -sum(posQt, dims = 1) .* Xt
    return posQt .+ diagQt
end

function velo_step(P, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, delta_t, log_velocity, scale)
    ohXₜ = onehot(Xₜ)
    velocity = rate_constraint(tensor(ohXₜ), log_velocity, P.transform) .* scale
    newXₜ = CategoricalLikelihood(eltype(delta_t).(tensor(ohXₜ) .+ (delta_t .* velocity)))
    clamp!(tensor(newXₜ), 0, Inf) #Because one velo will be < 0 and a large step might push Xₜ < 0
    return rand(newXₜ)
end

step(P::DoobMatchingFlow, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, veloX̂₁::Flowfusion.Guide, s₁, s₂) = velo_step(P, Xₜ, s₂ .- s₁, veloX̂₁.H, expand(1 ./ onescale(P, s₁), ndims(veloX̂₁.H)))
step(P::DoobMatchingFlow, Xₜ::DiscreteState{<:AbstractArray{<:Signed}}, veloX̂₁, s₁, s₂) = velo_step(P, Xₜ, s₂ .- s₁, veloX̂₁, expand(1 ./ onescale(P, s₁), ndims(veloX̂₁)))

function cgm_dloss(P, Xt, X̂₁, doobX₁)
    Qt = P.transform(X̂₁)
    return sum((1 .- Xt) .* (Qt .- xlogy.(doobX₁, Qt)), dims = 1) #<- note, diagonals ignored; implicit zero sum
end

floss(P::Flowfusion.fbu(DoobMatchingFlow), Xt::Flowfusion.msu(DiscreteState), X̂₁, X₁::Guide, c) = Flowfusion.scaledmaskedmean(cgm_dloss(P, tensor(Xt), tensor(X̂₁), X₁.H), c, Flowfusion.getlmask(X₁))