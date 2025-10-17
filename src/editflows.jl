
struct EditFlow <: DiscreteIndelProcess
    k::Int                 # alphabet size (tokens 1..k)
    transform::Function    # maps unconstrained logits to positive rates
    κ::Function            # scheduler on [0,1] → [0,1] for bridge keep-probability
    padding_token::Int     # padding token id used in X1
    latent_token::Int      # latent placeholder used in Z0 / Zt
    bos_token::Int         # beginning-of-sequence token id (optional, unused by default)
end

EditFlow(k; transform = NNlib.softplus,
            κ = identity,
            padding_token::Int = k + 1,
            latent_token::Int = k + 2,
            bos_token::Int = 0) =
    EditFlow(k, transform, κ, padding_token, latent_token, bos_token)


@inline function pick_index(w::AbstractVector{<:Real})::Int
    # treat negatives as zero; assert we have some mass
    cs = cumsum(max.(w, zero(eltype(w))))
    s  = cs[end]
    @assert isfinite(s) && s > 0 "pick_index: all weights ≤ 0 or non-finite"
    u = rand() * s
    return searchsortedfirst(cs, u)  # 1..length(w)
end

function step(P::EditFlow,
              Xt::DiscreteState{<:AbstractArray{<:Signed}},
              hat,
              s1::Real, s2::Real)

    @assert ndims(Xt.state) == 1 "EditFlow.step only supports 1D DiscreteState"

    # Rates
    pins, psub, pdel = part_output(P, P.transform(hat))   # (K,n+1,B), (K,n,B), (1,n,B)
    ins = Array(pins[:, :, 1])                            # (K, n+1) or (K, n)
    sub = Array(psub[:, :, 1]) 
    del = vec(Array(pdel[1, :, 1]))                       # (n,)  <-- fixed

    K, n = size(sub, 1), size(sub, 2)
    @assert size(ins, 1) == K
    @assert length(del) == n

    # Ensure gaps shape (K, n+1)
    ins_gaps = if size(ins, 2) == n + 1
        ins
    elseif size(ins, 2) == n
        tmp = similar(ins, K, n + 1)
        @inbounds for s in 0:n
            pos = clamp(s, 1, n)
            @views tmp[:, s + 1] .= ins[:, pos]
        end
        tmp
    else
        error("EditFlow.step: bad ins size $(size(ins))")
    end

    dt = float(s2 - s1)
    x = collect(tensor(Xt))  # Vector{Int}

    # Forbid self-substitutions
    if n > 0
        current_mask = zeros(eltype(sub), size(sub))
        @inbounds for i in 1:n
            tok = x[i]
            if 1 ≤ tok ≤ K
                current_mask[tok, i] = 1
            end
        end
        sub .*= (1 .- current_mask)
    end

    # Optionally forbid editing BOS explicitly
    if n > 0 && x[1] == P.bos_token
        sub[:, 1] .= 0
        del[1] = 0
    end

    # ---- site events (delete/sub) ----
    to_delete = falses(n)
    sub_to    = zeros(Int, n)
    @inbounds for i in 1:n
        r_del = del[i]
        r_sub_total = sum(@view sub[:, i])
        r_tot = r_del + r_sub_total
        if r_tot > 0 && rand() < (1 - exp(-dt * r_tot))
            u = rand() * r_tot
            if u < r_del
                to_delete[i] = true
            elseif r_sub_total > 0
                sub_to[i] = pick_index(@view sub[:, i])
            end
        end
    end

    # ---- gap insertions (≤1 per gap) ----
    ins_tok = fill(0, n + 1)
    start_gap = (n > 0 && x[1] == P.bos_token) ? 1 : 0
    @inbounds for s in start_gap:n
        r_ins_total = sum(@view ins_gaps[:, s + 1])
        #println("r_ins_total", ins_gaps[:, s + 1])
        if r_ins_total > 0 && rand() < (1 - exp(-dt * r_ins_total))
            ins_tok[s + 1] = pick_index(@view ins_gaps[:, s + 1])
        end
    end

    # ---- build new sequence ----
    result = Int[]
    if ins_tok[1] != 0; push!(result, ins_tok[1]); end
    @inbounds for i in 1:n
        if !to_delete[i]
            a = (sub_to[i] == 0) ? x[i] : sub_to[i]
            push!(result, a)
        end
        if ins_tok[i + 1] != 0
            push!(result, ins_tok[i + 1])
        end
    end
    return DiscreteState(Xt.K, result)
end

"""
    bridge(P::EditFlow, X0::DiscreteState, X1::DiscreteState, t)

Latent-style bridge that selects each token of `X1` independently with probability κ(t)
and drops the rest, then returns the resulting subsequence (no padding). No special BOS handling.
"""
function bridge(P::EditFlow,
                X0::DiscreteState{<:AbstractArray{<:Signed}},
                X1::DiscreteState{<:AbstractArray{<:Signed}},
                t::Real)
    keep_prob = clamp(P.κ(t), 0, 1)
    x1 = collect(tensor(X1))

    # Z0: replace non-padding tokens with latent; preserve padding
    z0 = Vector{Int}(undef, length(x1))
    @inbounds for i in eachindex(x1)
        tok = x1[i]
        if tok == P.padding_token
            z0[i] = tok
        else
            z0[i] = P.latent_token
        end
    end
    z1 = x1

    # Mix via scheduler mask: Zt = if rand<κ(t) then Z1 else Z0
    zt = similar(z0)
    @inbounds for i in eachindex(z0)
        zt[i] = (rand() < keep_prob) ? z1[i] : z0[i]
    end

    # Remove latent and padding
    global result = Int[]
    for tok in zt
        if tok != P.latent_token && tok != P.padding_token
            push!(result, tok)
        end
    end
    error("EditFlow bridge: not implemented for this configuration")
    return DiscreteState(P.k, result)
end

function bridge(P::EditFlow,
                X0s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                tvec::AbstractVector{<:Real})
    @assert length(X0s) == length(X1s) == length(tvec)
    return [bridge(P, X0s[i], X1s[i], tvec[i]) for i in eachindex(X0s)]
end

"""
    bridge_batched(P::EditFlow, X0s::Vector{<:DiscreteState}, X1s::Vector{<:DiscreteState}, tvec; keep_padding=true)

Applies `bridge` per-sample, then pads/semi-batches the resulting sequences with `Flowfusion.batch`.
This mirrors `remove_and_pad_concise` behavior at the batch level and returns a `MaskedState`.
"""
function bridge_batched(P::EditFlow,
                        X0s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                        X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                        tvec::AbstractVector{<:Real})
    xs = bridge(P, X0s, X1s, tvec)
    return Flowfusion.batch(xs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Guide helpers for EditFlow
# ─────────────────────────────────────────────────────────────────────────────

"""
    Guide(P::EditFlow, M)

Wrap a combined-rate tensor `M` of shape (2K+1, n[, B]) into a Guide with
NamedTuple fields (sub::(K,n[,B]), del::(1,n[,B]), ins::(K,n+1[,B])).
Insertion rates are mapped from positions to gaps by assigning gap s to position clamp(s,1,n).
"""
function Guide(P::EditFlow, M::AbstractArray)
    K = P.k
    nd = ndims(M)
    @assert nd in (2,3)
    if nd == 2
        K2, n = size(M)
        B = 1
        M3 = reshape(M, K2, n, 1)
    else
        K2, n, B = size(M)
        M3 = M
    end
    @assert K2 == 2*K + 1
    S = eltype(M3)
    sub = similar(M3, S, K, n, B)
    del = similar(M3, S, 1, n, B)
    ins = zeros(S, K, n + 1, B)
    @views sub[:, :, :] .= M3[K+1:2K, :, :]
    @views del[1, :, :] .= M3[2K+1, :, :]
    # map position-based insertion to gaps
    for s in 0:n
        pos = clamp(s, 1, n)
        @views ins[:, s+1, :] .= M3[1:K, pos, :]
    end
    return Flowfusion.Guide((sub=sub, del=del, ins=ins))
end

function part_output(P::EditFlow, M::AbstractArray)
    K = P.k
    ins = M[1:K,:,:]
    sub = M[K+1:2K,:,:]
    del = M[2K+1:2K+1,:,:]
    return ins, sub, del
end


#=
function edit_loss(P::EditFlow,
                   M::AbstractArray,
                   transition_mask::AbstractArray,
                   edit_multiplier::AbstractArray,
                   scheduler_scaling;
                   op_mask=nothing,
                   eps=1e-8)
    R = P.transform(M)
    OM = isnothing(op_mask) ? one(eltype(R)) .* ones(eltype(R), size(R)) : op_mask
    term1 = sum(transition_mask .* (OM .* R); dims=(1,2))
    scl = reshape(scheduler_scaling, 1, 1, :)
    term2 = sum(scl .* edit_multiplier .* log.(R .+ eps); dims=(1,2))
    return mean(term1 .- term2)
end
=#
"""
    edit_loss(P::EditFlow, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=nothing, eps=1e-8)

Loss matching the reference: mean(sum(transition_mask .* (op_mask .* R)) - sum(scheduler_scaling .* edit_multiplier .* log R)),
where R = transform(M).
Shapes:
- M, transition_mask, edit_multiplier, op_mask: (2K+1, n, B)
- scheduler_scaling: (1, B) or (B,) broadcastable to (1,1,B)
"""
function edit_loss(P::EditFlow,
                   M, transition_mask, edit_multiplier, scheduler_scaling;
                   op_mask=nothing, eps=1e-8)

    R = P.transform(M)                              # must be >= 0
    # (A) Optional op mask to apply symmetrically
    OM = isnothing(op_mask) ? one(eltype(R)) : op_mask

    # (B) Sum of valid outgoing rates
    term1 = sum(transition_mask .* (OM .* R); dims=(1,2))

    # (C) Logs only of positive rates (avoid NaN/Inf)
    R_logsafe = max.(R, eltype(R)(eps))             # clamp BEFORE log
    logR = log.(R_logsafe)

    scl = reshape(scheduler_scaling, 1, 1, :)       # (1,1,B)
    term2 = sum(scl .* (edit_multiplier .* OM) .* logR; dims=(1,2))

    return mean(term1 .- term2)
end
#=
function edit_loss(P::EditFlow,
                   M::AbstractArray,
                   transition_mask::AbstractArray,
                   edit_multiplier::AbstractArray,
                   scheduler_scaling;
                   op_mask=nothing,
                   eps=1e-8)
    R = P.transform(M)
    OM = isnothing(op_mask) ? one(eltype(R)) : op_mask
    term1 = sum(transition_mask .* (OM .* R); dims=(1,2))
    scl = reshape(scheduler_scaling, 1, 1, :)
    epsT = eltype(R)(eps)
    term2 = sum(scl .* edit_multiplier .* log.(R .+ epsT); dims=(1,2))
    return mean(term1 .- term2)
end
=#
"""
    combine_rates(P::EditFlow, G::Guide) -> M

Convert a Guide with fields (sub, del, ins) to combined shape (2K+1, n[, B])
by dropping gap 0 and mapping gaps 1..n to positions 1..n.
"""
function combine_rates(P::EditFlow, G::Guide)
    K = P.k
    sub = G.H.sub
    del = G.H.del
    ins = G.H.ins
    nd = ndims(sub)
    if nd == 2
        K2, n = size(sub)
        B = 1
    else
        K2, n, B = size(sub)
    end
    @assert size(del, 2) == n
    @assert size(ins, 2) == n + 1
    S = eltype(sub)
    M = zeros(S, 2*K + 1, n, B)
    @views M[K+1:2K, :, :] .= sub
    @views M[2K+1, :, :] .= del[1, :, :]
    # map gaps 1..n to positions 1..n; ignore gap 0
    @views M[1:K, :, :] .= ins[:, 2:end, :]
    return M
end

"""
    floss(P::EditFlow, Xt, X̂₁, G, scheduler_scaling, transition_mask, edit_multiplier; op_mask=nothing, eps=1e-8)

Convenience wrapper: accepts either combined `X̂₁` of shape (2K+1,n[,B]) or a Guide payload,
and computes `edit_loss` with provided masks and scaling.
"""
# function floss(P::EditFlow,
#                Xt::MaskedState{<:DiscreteState},
#                X̂₁,
#                G::Guide,
#                scheduler_scaling,
#                transition_mask,
#                edit_multiplier;
#                op_mask=nothing,
#                eps=1e-8)
#     M = X̂₁ isa AbstractArray ? X̂₁ : combine_rates(P, Flowfusion.Guide(X̂₁))
#     return edit_loss(P, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=op_mask, eps=eps)
# end

# ─────────────────────────────────────────────────────────────────────────────
# Remaining-edits and masks (from reference training loop), batched
# ─────────────────────────────────────────────────────────────────────────────

"""
    remove_and_pad_concise(Zt, latent_token, padding_token) -> Xt

Drop `latent_token` from each column and pad with `padding_token` to the max column length.
Returns a matrix Xt of shape (xt_length, B).
"""
function remove_and_pad_concise(Zt::AbstractMatrix{<:Integer}, latent_token::Integer, padding_token::Integer)
    filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    max_len = maximum(length, filtered_cols)
    return hcat(map(col -> vcat(col, fill(padding_token, max_len - length(col))), filtered_cols)...)
end

"""
    remaining_edits_dense(P, Zt, Z1) -> E

Compute dense edit-multiplier tensor E of shape (2K+1, xt_length, B) from latent Zt and target Z1.
Row mapping: 1..K insertions, K+1..2K substitutions to that token, 2K+1 deletions.
"""
# function remaining_edits_dense(P::EditFlow, Zt::AbstractMatrix{<:Integer}, Z1::AbstractMatrix{<:Integer})
#     @assert size(Zt) == size(Z1)
#     tokens = P.k
#     padding_token = P.padding_token
#     latent_token = P.latent_token
#     (_, B) = size(Z1)

#     # positions within filtered Xt (cumulative non-(latent|padding))
#     pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)
#     filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
#     xt_len = maximum(length, filtered_cols)

#     E = zeros(Float32, 2*tokens + 1, xt_len, B)

#     # Insertions: positions where Zt has latent and Z1 has real token
#     inserts = Z1 .* (Zt .== latent_token)
#     for I in findall(!iszero, inserts)
#         tok = inserts[I]
#         col = pos[I]
#         col = clamp(col, 1, xt_len)
#         E[tok, col, I[2]] += 1
#     end

#     # Substitutions: Zt != latent, Z1 != latent, Z1 != Zt
#     subs_mask = (Zt .!= latent_token) .& (Z1 .!= latent_token) .& (Z1 .!= Zt)
#     subs = Z1 .* subs_mask
#     for I in findall(!iszero, subs)
#         tok = subs[I]
#         col = pos[I]
#         col = clamp(col, 1, xt_len)
#         E[tokens + tok, col, I[2]] += 1
#     end

#     # Deletions: Z1 is latent (i.e., target removes this position)
#     dels = (Z1 .== latent_token)
#     for I in findall(dels)
#         col = pos[I]
#         col = clamp(col, 1, xt_len)
#         E[2*tokens + 1, col, I[2]] += 1
#     end

#     return E
# end

#= My implementation of remaining_edits_dense
function remaining_edits(P::EditFlow, Zt::AbstractMatrix{Int}, Z1::AbstractMatrix{Int}; dense=false)
    latent_token = P.latent_token
    padding_token = P.padding_token
    tokens = P.k

    (_, batch_size) = size(Z1)

    filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    batch_length = maximum(length, filtered_cols) 
    
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)
    #Insert 
    inserts = Z1.*(Zt .== latent_token)
    insert_edits = zeros(Float32, (tokens, batch_length, batch_size))
    insert_indices = findall(!iszero, inserts)
    insert_cols_to_update = pos[insert_indices]
    insert_rows_to_update = inserts[insert_indices]
    insert_samples_to_update = [idx[2] for idx in insert_indices]
    for i in 1:length(insert_rows_to_update)
        insert_edits[insert_rows_to_update[i], insert_cols_to_update[i], insert_samples_to_update[i]] += 1
    end
    dense_inserts = (insert_rows_to_update, insert_cols_to_update, insert_samples_to_update)

    #Substitution
    a = Zt .!= latent_token
    b = Z1 .!= latent_token
    c = Z1 .!= Zt
    subs = Z1.*(a .& b .& c)
    sub_edits = zeros(Float32, (tokens, batch_length, batch_size))
    sub_indices = findall(!iszero, subs)
    sub_cols_to_update = pos[sub_indices]
    sub_rows_to_update = subs[sub_indices]
    sub_samples_to_update = [idx[2] for idx in sub_indices]
    for i in 1:length(sub_rows_to_update)
        sub_edits[sub_rows_to_update[i], sub_cols_to_update[i], sub_samples_to_update[i]] = 1
    end
    dense_subs = (sub_rows_to_update .+ tokens, sub_cols_to_update, sub_samples_to_update)

    #Del
    dels = Z1 .== latent_token
    del_edits = zeros(Float32, (1, batch_length, batch_size))
    del_indices = findall(!iszero, dels)
    del_cols_to_update = pos[del_indices]
    del_samples_to_update = [idx[2] for idx in del_indices]
    for i in 1:length(del_cols_to_update)
        del_edits[1, del_cols_to_update[i], del_samples_to_update[i]] = 1
    end
    dense_dels = ((2*tokens+1).*ones(Int64, length(del_cols_to_update)), del_cols_to_update, del_samples_to_update) 

    if dense == true
        return (dense_inserts, dense_subs, dense_dels)
    else
        return vcat(insert_edits, sub_edits, del_edits)
    end

end
=#


#=
function remaining_edits(P::EditFlow, Zt::AbstractMatrix{Int}, Z1::AbstractMatrix{Int}; dense=false)
    latent_token   = P.latent_token
    padding_token  = P.padding_token
    tokens         = P.k
    (_, batch_size) = size(Z1)

    # Xt length = max over columns after removing latent
    filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    batch_length  = maximum(length, filtered_cols)

    # Position indices within Xt (count non-(latent|padding))
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)

    # -------------------- Inserts: Zt==latent, Z1 in 1..K --------------------
    insert_mask = (Zt .== latent_token) .& (Z1 .>= 1) .& (Z1 .<= tokens)
    insert_edits = zeros(Float32, (tokens, batch_length, batch_size))
    insert_indices = findall(insert_mask)
    insert_cols_to_update = clamp.(pos[insert_mask], 1, batch_length)
    insert_rows_to_update = Z1[insert_mask]                     # 1..K
    insert_samples_to_update = [idx[2] for idx in insert_indices]
    @inbounds for i in 1:length(insert_rows_to_update)
        insert_edits[insert_rows_to_update[i], insert_cols_to_update[i], insert_samples_to_update[i]] += 1
    end
    dense_inserts = (insert_rows_to_update, insert_cols_to_update, insert_samples_to_update)

    # ------------- Substitutions: Zt∉{latent,pad}, Z1 in 1..K, Z1≠Zt -------------
    sub_mask = (Zt .!= latent_token) .& (Zt .!= padding_token) .&
               (Z1 .>= 1) .& (Z1 .<= tokens) .& (Z1 .!= Zt)
    sub_edits = zeros(Float32, (tokens, batch_length, batch_size))
    sub_indices = findall(sub_mask)
    sub_cols_to_update = clamp.(pos[sub_mask], 1, batch_length)
    sub_rows_to_update = Z1[sub_mask]                           # 1..K
    sub_samples_to_update = [idx[2] for idx in sub_indices]
    @inbounds for i in 1:length(sub_rows_to_update)
        sub_edits[sub_rows_to_update[i], sub_cols_to_update[i], sub_samples_to_update[i]] = 1
    end
    dense_subs = (sub_rows_to_update .+ tokens, sub_cols_to_update, sub_samples_to_update)

    # ------------------------ Deletions: Z1==latent ------------------------
    dels = (Z1 .== latent_token)
    del_edits = zeros(Float32, (1, batch_length, batch_size))
    del_indices = findall(dels)
    del_cols_to_update = clamp.(pos[dels], 1, batch_length)
    del_samples_to_update = [idx[2] for idx in del_indices]
    @inbounds for i in 1:length(del_cols_to_update)
        del_edits[1, del_cols_to_update[i], del_samples_to_update[i]] = 1
    end
    dense_dels = ((2*tokens+1) .* ones(Int, length(del_cols_to_update)), del_cols_to_update, del_samples_to_update)

    return dense ? (dense_inserts, dense_subs, dense_dels) : vcat(insert_edits, sub_edits, del_edits)
end
=#

# Safe override for remaining_edits to avoid out-of-bounds row indexing
# Ensures only 1..P.k token ids are used as row indices and batch length matches pos
function remaining_edits(P::FF.EditFlow, Zt::AbstractMatrix{Int}, Z1::AbstractMatrix{Int}; dense::Bool=false)
    @assert size(Zt) == size(Z1)
    L, B = size(Z1)

    latent_token  = P.latent_token
    padding_token = P.padding_token
    tokens        = P.k

    # Position index per (i,j); pos is (L,B)
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)
    batch_length = maximum(vec(pos[end, :]))

    # Inserts: positions where Zt is latent; only valid row ids 1..tokens
    ins_mask = (Zt .== latent_token) .& (1 .<= Z1 .<= tokens)
    insert_indices = findall(ins_mask)
    insert_rows_to_update = Z1[insert_indices]
    insert_cols_to_update = pos[insert_indices]
    insert_samples_to_update = ntuple(_ -> 0, 0); insert_samples_to_update = [I[2] for I in insert_indices]

    insert_edits = zeros(Float32, (tokens, batch_length, B))
    @inbounds for i in 1:length(insert_rows_to_update)
        insert_edits[insert_rows_to_update[i], insert_cols_to_update[i], insert_samples_to_update[i]] += 1f0
    end
    dense_inserts = (insert_rows_to_update, insert_cols_to_update, insert_samples_to_update)

    # Substitutions: valid tokens, different from latent and changed from Zt
    sub_mask = (Zt .!= latent_token) .& (Z1 .!= latent_token) .& (Z1 .!= Zt) .& (1 .<= Z1 .<= tokens)
    sub_indices = findall(sub_mask)
    sub_rows_to_update = Z1[sub_indices]
    sub_cols_to_update = pos[sub_indices]
    sub_samples_to_update = ntuple(_ -> 0, 0); sub_samples_to_update = [I[2] for I in sub_indices]

    sub_edits = zeros(Float32, (tokens, batch_length, B))
    @inbounds for i in 1:length(sub_rows_to_update)
        sub_edits[sub_rows_to_update[i], sub_cols_to_update[i], sub_samples_to_update[i]] = 1f0
    end
    dense_subs = (sub_rows_to_update .+ tokens, sub_cols_to_update, sub_samples_to_update)

    # Deletions: latent in Z1
    del_mask = (Z1 .== latent_token)
    del_indices = findall(del_mask)
    del_cols_to_update = pos[del_indices]
    del_samples_to_update = ntuple(_ -> 0, 0); del_samples_to_update = [I[2] for I in del_indices]

    del_edits = zeros(Float32, (1, batch_length, B))
    @inbounds for i in 1:length(del_cols_to_update)
        del_edits[1, del_cols_to_update[i], del_samples_to_update[i]] = 1f0
    end
    dense_dels = ((2*tokens+1).*ones(Int64, length(del_cols_to_update)), del_cols_to_update, del_samples_to_update)

    if dense
        return (dense_inserts, dense_subs, dense_dels)
    else
        return vcat(insert_edits, sub_edits, del_edits)
    end
end


"""
getlmask(P::EditFlow, Xt::AbstractMatrix{<:Integer})

Build lmask of shape (xt_length, B) from padded Xt.
"""
function getlmask(P::EditFlow, Xt::AbstractMatrix{<:Integer})
    padding_token = P.padding_token
    return Xt .!= padding_token
end
"""
    transition_mask_from_Xt(P, Xt) -> T

Build transition mask T of shape (2K+1, xt_length, B) from padded Xt.
Zeros rows corresponding to self-substitutions and fully zeros columns that are padding.
"""
function transition_mask_from_Xt(P::EditFlow, Xt::AbstractMatrix{<:Integer})
    tokens = P.k
    pad = P.padding_token
    xt_len, B = size(Xt)
    T = ones(Float32, 2*tokens + 1, xt_len, B)
    for c in 1:B
        for i in 1:xt_len
            x = Xt[i, c]
            @assert x != P.latent_token
            if x == pad
                T[:, i, c] .= 0
            elseif x == P.bos_token
                @assert i == 1 #should only be BOS at position 1
                T[tokens+1:2*tokens+1, i, c] .= 0
            else
                # forbid sub-to-current-token only for valid tokens 1..K
                if 1 <= x <= tokens
                    T[tokens + x, i, c] = 0
                end
            end
        end
    end
    return T
end




#=
#= 
EditFlow: a simple edit-based discrete flow with insert/substitute/delete events.

Intent:
- Provide a generic edit-process akin to Poisson–Indel but driven by model-provided instantaneous hazards.
- step(EditFlow, Xt, hat, s1, s2) samples at most one event per site/gap in [s1,s2] using thinning.

Guide payload convention (token-first):
- hat.ins :: (K, n+1, B?) per-gap insertion hazards (per token 1..K; gap indices 0..n)
- hat.sub :: (K, n,   B?) per-site substitution hazards (no self-sub, caller/model should zero it if desired)
- hat.del :: (1, n,   B?) per-site deletion hazards

The EditFlow does not impose Doob-alignment logic; it is model-agnostic and only samples edits from hazards.
This makes it suitable for experimenting with latent bridges like the provided script, while matching the
Flowfusion Guide/step interface so it can be used in gen.
=#

struct EditFlow <: DiscreteIndelProcess
    k::Int                 # alphabet size (tokens 1..k)
    transform::Function    # maps unconstrained logits to positive rates
    κ::Function            # scheduler on [0,1] → [0,1] for bridge keep-probability
    padding_token::Int     # padding token id used in X1
    latent_token::Int      # latent placeholder used in Z0 / Zt
end

EditFlow(k; transform = NNlib.softplus, κ = identity, padding_token::Int = k, latent_token::Int = k + 1) =
    EditFlow(k, transform, κ, padding_token, latent_token)

"""
    step(P::EditFlow, Xt::DiscreteState, hat, s1, s2)

Takes a short stochastic step under instantaneous hazards in `hat`:
- sub :: (K, n) per-position substitution hazards (self-sub not enforced here)
- del :: (n,) deletion hazards per position
- ins :: (K, n+1) per-gap insertion hazards

At most one event per site/gap is applied using Poisson thinning with probability 1 - exp(-dt * rate_total).
"""
function step(P::EditFlow,
              Xt::DiscreteState{<:AbstractArray{<:Signed}},
              hat,
              s1::Real, s2::Real)
    @assert ndims(Xt.state) == 1 "EditFlow.step only supports 1D DiscreteState"
    # Transform hazards
    sub = Array(P.transform(hat.sub)[:, :, 1])  # (K, n)
    del = vec(Array(P.transform(hat.del))[1, :, 1])  # (n,)
    ins = Array(P.transform(hat.ins)[:, :, 1])  # (K, n+1)

    K, n = size(sub, 1), size(sub, 2)
    @assert length(del) == n
    @assert size(ins, 1) == K && size(ins, 2) == n + 1

    dt = float(s2 - s1)
    x = collect(tensor(Xt))  # Vector{Int}

    # Forbid self-substitutions w.r.t. current tokens
    if n > 0
        current_mask = tensor(onehot(Xt))[:,:,1]
        sub .= sub .* (1 .- current_mask)
    end

    # Site events: delete/substitute
    to_delete = falses(n)
    sub_to    = zeros(Int, n)   # 0 => no substitution; otherwise token id
    for i in 1:n
        r_del = del[i]
        r_sub_total = sum(@view sub[:, i])
        r_tot = r_del + r_sub_total
        if r_tot > 0
            if rand() < (1 - exp(-dt * r_tot))
                u = rand() * r_tot
                if u < r_del
                    to_delete[i] = true
                else
                    u2 = u - r_del
                    acc = 0.0
                    chosen = 0
                    @inbounds for tok in 1:K
                        acc += sub[tok, i]
                        if u2 <= acc
                            chosen = tok
                            break
                        end
                    end
                    # fall back if needed
                    chosen == 0 && (chosen = 1)
                    sub_to[i] = chosen
                end
            end
        end
    end

    # Gap insertions: propose at most one per gap
    ins_tok = fill(0, n + 1)  # 0 => none; otherwise token id
    for s in 0:n
        r_ins_total = sum(@view ins[:, s + 1])
        if r_ins_total > 0
            if rand() < (1 - exp(-dt * r_ins_total))
                u = rand() * r_ins_total
                acc = 0.0
                chosen = 0
                @inbounds for tok in 1:K
                    acc += ins[tok, s + 1]
                    if u <= acc
                        chosen = tok
                        break
                    end
                end
                chosen == 0 && (chosen = 1)
                ins_tok[s + 1] = chosen
            end
        end
    end

    # Build new sequence
    result = Int[]
    if ins_tok[1] != 0
        push!(result, ins_tok[1])
    end
    for i in 1:n
        if !to_delete[i]
            a = sub_to[i] == 0 ? x[i] : sub_to[i]
            push!(result, a)
        end
        if ins_tok[i + 1] != 0
            push!(result, ins_tok[i + 1])
        end
    end

    return DiscreteState(Xt.K, result)
end

"""
    bridge(P::EditFlow, X0::DiscreteState, X1::DiscreteState, t)

Latent-style bridge that selects each token of `X1` independently with probability κ(t)
and drops the rest, then returns the resulting subsequence (no padding). No special BOS handling.
"""
function bridge(P::EditFlow,
                X0::DiscreteState{<:AbstractArray{<:Signed}},
                X1::DiscreteState{<:AbstractArray{<:Signed}},
                t::Real)
    keep_prob = clamp(P.κ(t), 0, 1)
    x1 = collect(tensor(X1))

    # Z0: replace non-padding tokens with latent; preserve padding
    z0 = Vector{Int}(undef, length(x1))
    @inbounds for i in eachindex(x1)
        tok = x1[i]
        if tok == P.padding_token
            z0[i] = tok
        else
            z0[i] = P.latent_token
        end
    end
    z1 = x1

    # Mix via scheduler mask: Zt = if rand<κ(t) then Z1 else Z0
    zt = similar(z0)
    @inbounds for i in eachindex(z0)
        zt[i] = (rand() < keep_prob) ? z1[i] : z0[i]
    end

    # Remove latent and padding
    result = Int[]
    for tok in zt
        if tok != P.latent_token && tok != P.padding_token
            push!(result, tok)
        end
    end

    return DiscreteState(P.k, result)
end

function bridge(P::EditFlow,
                X0s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                tvec::AbstractVector{<:Real})
    @assert length(X0s) == length(X1s) == length(tvec)
    return [bridge(P, X0s[i], X1s[i], tvec[i]) for i in eachindex(X0s)]
end

"""
    bridge_batched(P::EditFlow, X0s::Vector{<:DiscreteState}, X1s::Vector{<:DiscreteState}, tvec; keep_padding=true)

Applies `bridge` per-sample, then pads/semi-batches the resulting sequences with `Flowfusion.batch`.
This mirrors `remove_and_pad_concise` behavior at the batch level and returns a `MaskedState`.
"""
function bridge_batched(P::EditFlow,
                        X0s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                        X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                        tvec::AbstractVector{<:Real})
    xs = bridge(P, X0s, X1s, tvec)
    return Flowfusion.batch(xs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Guide helpers for EditFlow
# ─────────────────────────────────────────────────────────────────────────────

"""
    Guide(P::EditFlow, M)

Wrap a combined-rate tensor `M` of shape (2K+1, n[, B]) into a Guide with
NamedTuple fields (sub::(K,n[,B]), del::(1,n[,B]), ins::(K,n+1[,B])).
Insertion rates are mapped from positions to gaps by assigning gap s to position clamp(s,1,n).
"""
function Guide(P::EditFlow, M::AbstractArray)
    K = P.k
    nd = ndims(M)
    @assert nd in (2,3)
    if nd == 2
        K2, n = size(M)
        B = 1
        M3 = reshape(M, K2, n, 1)
    else
        K2, n, B = size(M)
        M3 = M
    end
    @assert K2 == 2*K + 1
    S = eltype(M3)
    sub = similar(M3, S, K, n, B)
    del = similar(M3, S, 1, n, B)
    ins = zeros(S, K, n + 1, B)
    @views sub[:, :, :] .= M3[K+1:2K, :, :]
    @views del[1, :, :] .= M3[2K+1, :, :]
    # map position-based insertion to gaps
    for s in 0:n
        pos = clamp(s, 1, n)
        @views ins[:, s+1, :] .= M3[1:K, pos, :]
    end
    return Flowfusion.Guide((sub=sub, del=del, ins=ins))
end

"""
    edit_loss(P::EditFlow, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=nothing, eps=1e-8)

Loss matching the reference: mean(sum(transition_mask .* (op_mask .* R)) - sum(scheduler_scaling .* edit_multiplier .* log R)),
where R = transform(M).
Shapes:
- M, transition_mask, edit_multiplier, op_mask: (2K+1, n, B)
- scheduler_scaling: (1, B) or (B,) broadcastable to (1,1,B)
"""
function edit_loss(P::EditFlow,
                   M::AbstractArray,
                   transition_mask::AbstractArray,
                   edit_multiplier::AbstractArray,
                   scheduler_scaling;
                   op_mask=nothing,
                   eps=1e-8)
    R = P.transform(M)
    OM = isnothing(op_mask) ? one(eltype(R)) .* ones(eltype(R), size(R)) : op_mask
    term1 = sum(transition_mask .* (OM .* R); dims=(1,2))
    scl = reshape(scheduler_scaling, 1, 1, :)
    term2 = sum(scl .* edit_multiplier .* log.(R .+ eps); dims=(1,2))
    return mean(term1 .- term2)
end

"""
    combine_rates(P::EditFlow, G::Guide) -> M

Convert a Guide with fields (sub, del, ins) to combined shape (2K+1, n[, B])
by dropping gap 0 and mapping gaps 1..n to positions 1..n.
"""
function combine_rates(P::EditFlow, G::Guide)
    K = P.k
    sub = G.H.sub
    del = G.H.del
    ins = G.H.ins
    nd = ndims(sub)
    if nd == 2
        K2, n = size(sub)
        B = 1
    else
        K2, n, B = size(sub)
    end
    @assert size(del, 2) == n
    @assert size(ins, 2) == n + 1
    S = eltype(sub)
    M = zeros(S, 2*K + 1, n, B)
    @views M[K+1:2K, :, :] .= sub
    @views M[2K+1, :, :] .= del[1, :, :]
    # map gaps 1..n to positions 1..n; ignore gap 0
    @views M[1:K, :, :] .= ins[:, 2:end, :]
    return M
end

"""
    floss(P::EditFlow, Xt, X̂₁, G, scheduler_scaling, transition_mask, edit_multiplier; op_mask=nothing, eps=1e-8)

Convenience wrapper: accepts either combined `X̂₁` of shape (2K+1,n[,B]) or a Guide payload,
and computes `edit_loss` with provided masks and scaling.
"""
function floss(P::EditFlow,
               Xt::MaskedState{<:DiscreteState},
               X̂₁,
               G::Guide,
               scheduler_scaling,
               transition_mask,
               edit_multiplier;
               op_mask=nothing,
               eps=1e-8)
    M = X̂₁ isa AbstractArray ? X̂₁ : combine_rates(P, Flowfusion.Guide(X̂₁))
    return edit_loss(P, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=op_mask, eps=eps)
end

# ─────────────────────────────────────────────────────────────────────────────
# Remaining-edits and masks (from reference training loop), batched
# ─────────────────────────────────────────────────────────────────────────────

"""
    remove_and_pad_concise(Zt, latent_token, padding_token) -> Xt

Drop `latent_token` from each column and pad with `padding_token` to the max column length.
Returns a matrix Xt of shape (xt_length, B).
"""
function remove_and_pad_concise(Zt::AbstractMatrix{<:Integer}, latent_token::Integer, padding_token::Integer)
    filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    max_len = maximum(length, filtered_cols)
    return hcat(map(col -> vcat(col, fill(padding_token, max_len - length(col))), filtered_cols)...)
end

"""
    remaining_edits_dense(P, Zt, Z1) -> E

Compute dense edit-multiplier tensor E of shape (2K+1, xt_length, B) from latent Zt and target Z1.
Row mapping: 1..K insertions, K+1..2K substitutions to that token, 2K+1 deletions.
"""
function remaining_edits_dense(P::EditFlow, Zt::AbstractMatrix{<:Integer}, Z1::AbstractMatrix{<:Integer})
    @assert size(Zt) == size(Z1)
    tokens = P.k
    padding_token = P.padding_token
    latent_token = P.latent_token
    (_, B) = size(Z1)

    # positions within filtered Xt (cumulative non-(latent|padding))
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)
    filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    xt_len = maximum(length, filtered_cols)

    E = zeros(Float32, 2*tokens + 1, xt_len, B)

    # Insertions: positions where Zt has latent and Z1 has real token
    inserts = Z1 .* (Zt .== latent_token)
    for (r, c) in zip(findall(!iszero, inserts)...)
        tok = inserts[r, c]
        col = pos[r, c]
        E[tok, col, c] += 1
    end

    # Substitutions: Zt != latent, Z1 != latent, Z1 != Zt
    subs_mask = (Zt .!= latent_token) .& (Z1 .!= latent_token) .& (Z1 .!= Zt)
    subs = Z1 .* subs_mask
    for (r, c) in zip(findall(!iszero, subs)...)
        tok = subs[r, c]
        col = pos[r, c]
        E[tokens + tok, col, c] += 1
    end

    # Deletions: Z1 is latent (i.e., target removes this position)
    dels = (Z1 .== latent_token)
    for (r, c) in zip(findall(dels)...)
        col = pos[r, c]
        E[2*tokens + 1, col, c] += 1
    end

    return E
end

"""
    transition_mask_from_Xt(P, Xt) -> T

Build transition mask T of shape (2K+1, xt_length, B) from padded Xt.
Zeros rows corresponding to self-substitutions and fully zeros columns that are padding.
"""
function transition_mask_from_Xt(P::EditFlow, Xt::AbstractMatrix{<:Integer})
    tokens = P.k
    pad = P.padding_token
    xt_len, B = size(Xt)
    T = ones(Float32, 2*tokens + 1, xt_len, B)
    for c in 1:B
        for i in 1:xt_len
            x = Xt[i, c]
            if x == pad
                T[:, i, c] .= 0
            else
                # forbid sub-to-current-token
                T[tokens + x, i, c] = 0
            end
        end
    end
    return T
end


=#