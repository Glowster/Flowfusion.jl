using Test
using Flowfusion

@testset "EditFlow bridge latent masking" begin
    # tokens: 1..K, with bos=1, padding=K, latent=K+1
    K = 6
    bos = 1
    pad = K
    lat = K + 1
    κ(t) = t  # linearly increases keep prob
    P = EditFlow(K; κ=κ, bos_token=bos, padding_token=pad, latent_token=lat)

    # X0 has BOS only
    X0 = DiscreteState(K, [bos])
    # X1 is padded sequence: bos, real tokens, padding tail
    x1 = [bos, 2, 3, 4, pad, pad]
    X1 = DiscreteState(K, x1)

    # t=0: keep_prob=0 ⇒ only special tokens remain after masking+removal ⇒ [bos]
    Xt0 = Flowfusion.bridge(P, X0, X1, 0.0)
    @test tensor(Xt0) == [bos]

    # t=1: keep_prob=1 ⇒ all non-padding tokens from X1 kept (bos,2,3,4)
    Xt1 = Flowfusion.bridge(P, X0, X1, 1.0)
    @test tensor(Xt1) == [bos, 2, 3, 4]

    # t in (0,1): stochastic mixture; ensure BOS present and no padding/latent
    Xt5 = Flowfusion.bridge(P, X0, X1, 0.5)
    seq = tensor(Xt5)
    @test length(seq) >= 1
    @test seq[1] == bos
    @test all(tok != pad for tok in seq)
    @test all(tok != lat for tok in seq)
    @test all(1 <= tok <= K for tok in seq)
end

@testset "EditFlow step shape sanity" begin
    K = 5
    P = EditFlow(K)
    Xt = DiscreteState(K, [1,2,3])
    n = length(tensor(Xt))
    sub = zeros(Float64, K, n)
    del = zeros(Float64, 1, n)
    ins = zeros(Float64, K, n+1)
    guide = Flowfusion.Guide((sub=sub, del=del, ins=ins))
    Y = Flowfusion.step(P, Xt, guide, 0.0, 0.0)
    @test tensor(Y) == tensor(Xt)
end

@testset "EditFlow bridge_batched pads and masks" begin
    K = 6
    bos = 1
    pad = K
    lat = K + 1
    P = EditFlow(K; κ = t->t, bos_token=bos, padding_token=pad, latent_token=lat)

    # two samples with different X1 lengths
    X0s = [DiscreteState(K, [bos]), DiscreteState(K, [bos])]
    X1s = [DiscreteState(K, [bos,2,3,pad]), DiscreteState(K, [bos,4,pad,pad])]
    tvec = [1.0, 1.0]  # keep all non-padding tokens
    B = Flowfusion.bridge_batched(P, X0s, X1s, tvec)
    @test B isa MaskedState
    # Padding is dropped before batching; expect true content length (bos + non-pads)
    @test size(tensor(B.S), 1) == 3
    @test size(tensor(B.S), 2) == 2
    # Masks should be true up to each sample's length
    @test B.lmask[1:3, 1] == trues(3)
    @test B.lmask[1:2, 2] == trues(2)
end


