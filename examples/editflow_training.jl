# Sketch: training loop for EditFlow (mirrors PIP loop structure)
using Pkg
Pkg.activate(@__DIR__)
using Revise
using Random
using Statistics
using Adapt
using Functors
using Flux
import Flowfusion as FF

# Optional CUDA device helper (keeps code CPU-safe if CUDA not present)
const _gpu_enabled = try
    Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
    CUDA = Base.loaded_modules[Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")]
    CUDA.has_cuda()
catch
    false
end

to_dev(x) = _gpu_enabled ? Adapt.adapt(CUDA.CuArray, x) : x
to_cpu(x) = _gpu_enabled ? Adapt.adapt(Array, x) : x

# Your scheduler and derivative
default_k(t) = t
default_dk(t) = one(t)

"""
    make_minibatch(B; rng) -> x0s, x1s, ts

Placeholder minibatch maker: return
- x0s, x1s: Vector{FF.DiscreteState},
- ts: Vector{Float32} in [0,1].
Replace with your dataset pipeline.
"""
function make_minibatch(B::Int, P::FF.EditFlow; rng=Random.default_rng())
    K = P.k
    pad = P.padding_token
    bos = P.bos_token
    toks = collect(setdiff(1:K, (pad,)))
    x0s = Vector{FF.DiscreteState}(undef, B)
    x1s = Vector{FF.DiscreteState}(undef, B)
    for b in 1:B
        true_len = rand(rng, 4:8)
        real_tokens = rand(rng, toks, true_len) # exclude padding token
        seq = vcat(real_tokens, fill(pad, 8 - true_len))
        x1s[b] = FF.DiscreteState(K, seq)
        x0s[b] = FF.DiscreteState(K, [bos])
    end
    ts = rand(rng, Float32, B)
    return x0s, x1s, ts
end

"""
    latent_bridge_matrices(P, X1s, ts) -> (Z0, Z1, Zt)

Build Z0/Z1/Zt matrices as in the reference pipeline:
- Z1: padded X1 batch matrix
- Z0: turn non-padding tokens into latent; keep padding
- Zt: per-position Bernoulli(κ(t_b)) choose between Z1 and Z0 for column b
"""
function latent_bridge_matrices(P::FF.EditFlow,
                                X1s::Vector{<:FF.DiscreteState},
                                ts::AbstractVector)
    # Pad X1s to matrix (LxB)
    X1_ms = FF.batch(X1s)
    Z1 = FF.tensor(X1_ms)
    # Prepend BOS row to Z1
    Z1 = vcat(fill(P.bos_token, 1, size(Z1, 2)), Z1)
    # Construct Z0 (respect BOS row)
    Z0 = similar(Z1)
    @inbounds for j in axes(Z1, 2), i in axes(Z1, 1)
        tok = Z1[i, j]
        if i == 1
            Z0[i, j] = tok # keep BOS as is
        else
            Z0[i, j] = (tok == P.padding_token) ? tok : P.latent_token
        end
    end
    # Mix via κ(t)
    Zt = similar(Z1)
    @inbounds for j in axes(Z1, 2), i in axes(Z1, 1)
        keep = rand() < clamp(P.κ(ts[j]), 0, 1)
        # If i==1 (BOS row), always keep BOS from Z1
        Zt[i, j] = (i == 1) ? Z1[i, j] : (keep ? Z1[i, j] : Z0[i, j])
    end
    return Z0, Z1, Zt
end

"""
    train_editflow!(P, model; ...)

Minimal training loop for EditFlow.
Expects `model(ts, Xt_ms)` to return M of shape (2K+1, L, B) with positive logits/rates (will be passed through P.transform in the loss).
"""
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
    model = Functors.fmap(to_dev, model)
    opt_state = Flux.setup(Flux.Adam(lr), model)

    for epoch in 1:epochs
        for step in 1:steps_per_epoch
            # 1) Minibatch
            x0s, x1s, ts = make_minibatch(batch_size, P; rng=rng)

            # 2) Latent bridge (Z0/Z1/Zt), Xt padded, masks and multipliers
            Z0, Z1, Zt = latent_bridge_matrices(P, x1s, ts)
            Xt = FF.remove_and_pad_concise(Zt, P.latent_token, P.padding_token)
            lmask = FF.transition_mask_from_Xt(P, Xt)
            edit_multiplier = FF.remaining_edits(P, Zt, Z1)
            scheduler_scaling = dk.(ts) ./ (1 .- k.(ts))  # (B,)

            # 3) Build Xt MaskedState with lmask from padding
            #lmask = Xt .!= P.padding_token
            cmask = trues(size(lmask))
            Xt_ms = FF.MaskedState(FF.DiscreteState(P.k, Xt), cmask, lmask)
            
            # 4) Device
            ts_d    = to_dev(ts)
            Xt_ms_d = to_dev(Xt_ms)
            L_d     = to_dev(lmask)
            E_d     = to_dev(edit_multiplier)
            scl_d   = to_dev(reshape(Float32.(scheduler_scaling), 1, 1, :))

            # 5) Forward + loss + update
            loss, grad = Flux.withgradient(model) do m
                M = m(ts_d, Xt_ms_d)                  # (2K+1, L, B)
                FF.edit_loss(P, M, L_d, E_d, scl_d)
            end
            Flux.update!(opt_state, model, grad[1])

            if step % print_every == 0
                @info "train" epoch step loss=Float32(loss)
            end
        end
    end
    return model
end

# Example placeholder model (shapes only; replace with your transformer)
struct EditFlowDummyModel
    K::Int
end

function (m::EditFlowDummyModel)(ts, Xt_ms)
    # Xt_ms.S.state is (L, B) Int matrix; return random positive rates
    X = FF.tensor(Xt_ms.S)
    L, B = size(X)
    return abs.(randn(Float32, 2*m.K + 1, L, B)) .+ 1f-3
end

# Quick smoke run
K = 8
P = FF.EditFlow(K)
model = EditFlowDummyModel(K)
train_editflow!(P, model; epochs=1, steps_per_epoch=5, batch_size=16, lr=1f-3)



