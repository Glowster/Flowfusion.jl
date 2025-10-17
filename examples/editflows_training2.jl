using Pkg
Pkg.activate(@__DIR__)
using Revise

using Random
using Statistics
using Adapt
using Functors
using Flux
using Onion
using RandomFeatureMaps
using Zygote
import Flowfusion as FF

 

# Scheduler
default_k(t) = t
default_dk(t) = one(t)

# Toy minibatch: lengths 4..8, pad to 8 with P.padding_token; X0 is single BOS token
function make_minibatch(B::Int, P::FF.EditFlow; rng=Random.default_rng())
    K = P.k
    pad = P.padding_token
    bos = getfield(P, :bos_token) isa Int ? P.bos_token : K
    toks = collect(setdiff(1:K, (pad,)))
    x0s = Vector{FF.DiscreteState}(undef, B)
    x1s = Vector{FF.DiscreteState}(undef, B)
    for b in 1:B
        true_len = rand(rng, 1:3)
        real_tokens = rand(rng, toks, true_len)
        seq = vcat(real_tokens, fill(pad, 3 - true_len))
        x1s[b] = FF.DiscreteState(K, seq)
        x0s[b] = FF.DiscreteState(K, [bos])
    end
    ts = rand(rng, Float32, B)
    return x0s, x1s, ts
end

# Latent bridge with BOS preserved in Z1/Zt
function latent_bridge_matrices(P::FF.EditFlow, X1s::Vector{<:FF.DiscreteState}, ts::AbstractVector)
    X1_ms = FF.batch(X1s)
    Z1 = FF.tensor(X1_ms)
    # Prepend BOS row to Z1 if available
    if hasfield(typeof(P), :bos_token)
        Z1 = vcat(fill(P.bos_token, 1, size(Z1, 2)), Z1)
    end
    Z0 = similar(Z1)
    @inbounds for j in axes(Z1, 2), i in axes(Z1, 1)
        tok = Z1[i, j]
        if hasfield(typeof(P), :bos_token) && i == 1
            Z0[i, j] = tok
        else
            Z0[i, j] = (tok == P.padding_token) ? tok : P.latent_token
        end
    end
    Zt = similar(Z1)
    @inbounds for j in axes(Z1, 2), i in axes(Z1, 1)
        keep = rand() < clamp(P.κ(ts[j]), 0, 1)
        Zt[i, j] = (hasfield(typeof(P), :bos_token) && i == 1) ? Z1[i, j] : (keep ? Z1[i, j] : Z0[i, j])
    end
    return Z0, Z1, Zt
end

# Model backbone with Ada blocks; outputs combined M of shape (2K+1, L, B)
struct PIPModel{L}
    layers::L
end
Flux.@layer PIPModel

function PIPModel(; d=128, num_heads=8, nlayers=6, rff_dim=128, cond_dim=128, K::Int)
    # Support BOS=0 by shifting indices +1 into an embedding of size K+1
    embedding = Flux.Embedding(K + 1 => d)
    time_embed = Flux.Chain(RandomFourierFeatures(1 => rff_dim, 1.0f0), Dense(rff_dim => cond_dim))
    blocks = [Onion.AdaTransformerBlock(d, cond_dim, num_heads) for _ in 1:nlayers]
    head_combined = Dense(d => 2K + 1, bias=false)
    rope = RoPE(d ÷ num_heads, 4096)
    return PIPModel((; embedding, time_embed, blocks, head_combined, rope, K))
end


function (model::PIPModel)(t, Xt_ms)
    m = model.layers
    #@show Xt_ms
    X = FF.tensor(Xt_ms) #.S                    # (L, B) Int, may contain BOS=0
    #println(size(X))
    #@show typeof(X)
    X = ndims(X) == 1 ? reshape(X, :, 1) : X
    #println(size(X))
    #@show typeof(X)
    L, B = size(X)
    pmask = Zygote.@ignore FF.getlmask(Xt_ms)
    Xp = X .+ 1                               # shift 0..K -> 1..K+1
    H = m.embedding(Xp)                       # (d, L, B)
    t = ndims(t) == 0 ? fill(t, B) : t
    cond = m.time_embed(reshape(t, 1, B))     # (cond_dim, B)
    rope = Zygote.@ignore m.rope[1:L]
    for blk in m.blocks
        H = blk(H; cond, rope, kpad_mask=pmask)
    end
    M = m.head_combined(H)                    # (2K+1, L, B)
    return M 
end

function train_editflow!(P::FF.EditFlow,
                         model;
                         epochs::Int=1,
                         steps_per_epoch::Int=100,
                         batch_size::Int=64,
                         lr::Float32=1f-2,
                         seed::Int=42,
                         k::Function=default_k,
                         dk::Function=default_dk,
                         print_every::Int=25)

    rng = Random.MersenneTwister(seed)
    Random.seed!(seed)
    model = Functors.fmap(x -> x, model)
    opt_state = Flux.setup(Flux.Adam(lr), model)

    for epoch in 1:epochs
        for step in 1:steps_per_epoch
            # 1) Minibatch
            x0s, x1s, ts = make_minibatch(batch_size, P; rng=rng)

            # 2) Latent bridge
            Z0, Z1, Zt = latent_bridge_matrices(P, x1s, ts)
            Xt = FF.remove_and_pad_concise(Zt, P.latent_token, P.padding_token)
            transition_mask = FF.transition_mask_from_Xt(P, Xt)
            edit_multiplier = FF.remaining_edits(P, Zt, Z1)
            den = 1 .- k.(ts)
            den = max.(den, 1f-6)
            if any(den .<= 0)
                println("DEBUG den<=0: min=", minimum(den), " max=", maximum(den), " ts min/max=", (minimum(ts), maximum(ts)))
            end
            scheduler_scaling = dk.(ts) ./ den

            
            # 3) Masked state
            lmask = Xt .!= P.padding_token
            cmask = trues(size(lmask))
            Xt_ms = FF.MaskedState(FF.DiscreteState(P.k, Xt), cmask, lmask)
            
            # 4) Forward + loss + update
            # Debug forward pass outside AD so prints always execute
            println("time ts(min/max)=", (minimum(ts), maximum(ts)))
            println("DEBUG Xt(min/max)=", (minimum(Xt), maximum(Xt)))
            M_dbg = model(ts, Xt_ms)

            #println("DEBUG M has NaN/Inf: min/max=", (minimum(M_dbg), maximum(M_dbg)))
 
            R_dbg = P.transform(M_dbg)
            if any(R_dbg .<= 0)
                println("DEBUG rates<=0: count=", count(x -> x <= 0, R_dbg))
            end

            loss, grad = Flux.withgradient(model) do m
                M = m(ts, Xt_ms)                        # (2K+1, L, B)
                FF.edit_loss(P, M, transition_mask, edit_multiplier, reshape(Float32.(scheduler_scaling), 1, 1, :))
            end
            Flux.update!(opt_state, model, grad[1])

            if step % print_every == 0
                @info "train2" epoch step loss=Float32(loss)
            end
        end
    end
    return model
end

K = 8
P = FF.EditFlow(K; bos_token=0)
model = PIPModel(; d=128, num_heads=8, nlayers=4, rff_dim=128, cond_dim=128, K=K)
train_editflow!(P, model; epochs=1, steps_per_epoch=30, batch_size=16, lr=1f-3)



