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

function forward_positive_velocities(Xt::DiscreteState, P::HPiQ{T}) where T
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

function improved_forward_positive_velocities(Xt::DiscreteState, P::HPiQ{T}) where T
    (; tree, π) = P
    N = length(π)
    Xt = onehot(Xt) # Assume this is fast or input is already one-hot
    Q = zeros(Float64, size(Xt.state))
    all_nodes = PiNode[]
    ForwardBackward.get_all_nodes!(tree, all_nodes)
    batch_indices = onecold(Xt.state)
    
    # Get batch dimensions for CartesianIndices
    batch_dims = size(batch_indices)

    for node in all_nodes
        isnothing(node.leaf_indices) && continue
        idx = node.leaf_indices
        k = length(idx)
        k <= 1 && continue

        π_partition_view = view(π, idx)
        sum_π = sum(π_partition_view)
        isapprox(sum_π, 0.0) && continue

        # --- OPTIMIZATION 1: Pre-calculate updates and create a fast lookup map ---
        # A Dict provides O(1) average time for `haskey` and lookups.
        # This maps a global index to its local position (1:k) in the `idx` array.
        idx_to_local_map = Dict(j_global => i for (i, j_global) in enumerate(idx))
        
        # This vector of updates is calculated only once per node.
        node_updates = (node.u / sum_π) .* π_partition_view

        # --- OPTIMIZATION 2: Vectorize the innermost loop ---
        for I_tuple in CartesianIndices(batch_dims)
            # Create a CartesianIndex object. It's more idiomatic to pass the tuple.
            I = CartesianIndex(I_tuple) 
            b_idx = batch_indices[I]

            # Use the O(1) hash map lookup instead of an O(k) linear search.
            local_idx = get(idx_to_local_map, b_idx, 0) # Returns 0 if not found
            if local_idx > 0
                # Apply updates to the entire relevant slice of Q at once.
                # This replaces a loop of size k with a vectorized operation.
                Q_view = view(Q, idx, I)
                Q_view .+= node_updates

                # Correct the entry for the state itself (since j_global != batch_indices[I]).
                Q[b_idx, I] -= node_updates[local_idx]
            end
        end
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