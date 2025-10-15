#=
Poisson–Indel Process (PIP) bridge with Uniform Discrete substitutions (no explicit Q/P matrices).

Problem setup
- Alphabet: integers 1…K (K ≡ `p.k`).
- Substitution: Uniform-discrete continuous-time Markov chain with instantaneous off-diagonal rate α/K.
  Over time s, the transition kernel is:
    Pdiag(s) = e^{-α s} + (1 - e^{-α s})/K
    Poff(s)  = (1 - e^{-α s})/K
- Deletion: Each extant symbol survives a branch of length s with probability s_surv = e^{-μ s}.
- Insertion: Along a branch of length s, new symbols arrive via a Poisson process with rate λ.
  In the PIP construction, the expected number of descendants of a “virtual” immortal line is
    E[#insertions over s] = (λ/μ) * (1 - e^{-μ s}),
  and per-token probability mass is divided uniformly by K:
    Iins(s) = λ * ((1 - e^{-μ s})/μ) / K.

Bridge geometry
- We consider a node at time t ∈ [0,1] with sequence X_t (unknown).
- Left branch (to x0) has length t; right branch (to x1) has length s = 1 - t.
  Let s1 = e^{-μ t}, s2 = e^{-μ (1-t)}, and pow_sub(t) = e^{-α t} etc.

Two-branch alignment kernel (Uniform Discrete)
- For the pair (x0, x1) we define posterior proportions over three event types on the alignment grid:
  A: an x0-only letter (visible in x0, not in x1),
  B: an x1-only letter,
  R: a paired letter (visible in both x0 and x1).
- For R, the letter-level posterior depends on whether x0[i]==x1[j]:
  use P1(·; t) for the left branch and P2(·; 1-t) for the right branch.
- We compute unnormalized masses for A, B (aggregated over tokens) and R (split into “same token” vs “different”),
  and then normalize to probabilities pA, pB, pR_diag, pR_off. These are used by:
  - A forward DP computing the total alignment weight F[n+1,m+1].
  - A backward sampler drawing a particular alignment path (events).

Ancestral sequence sampling X_t
- Given the sampled alignment path, we sample each ancestral letter r ∈ {1,…,K} in O(1) time:
  - For an R event, P(r | x0[i], x1[j]) is proportional to P1(r→x0[i]) P2(r→x1[j]).
  - For A or B:
      There is a mixture “present-at-t” vs “inserted after t” with weights proportional to
        present: s_branch * (1 - s_other) / K,
        inserted: (λ/μ) * (1 - s_branch) / K.
      If present, sample r given the single observed descendant using P1 or P2.
  - Finally, we add “ghost” insertions in X_t with count ~ Poisson((λ/μ) (1 - s1) (1 - s2)),
    each placed uniformly among |X_t|+1 slots, and token ~ Uniform(1…K).

Single-branch inside–outside for Doob h-transform (hazard rates)
- On the right branch (length s = 1 - t), given current sequence X_t and leaf x1:
  - Forward-backward in log-space yields the current likelihood h_cur and local ratios used to form rates.
  - Deletion at position i: μ times posterior probability that this position is deleted on the remainder.
  - Substitution at i to token b: (α/K) times a ratio correcting for how b changes the inside–outside mass
    (no self-substitution; rate is zero for b==X_t[i]).
  - Insertion between positions s (0…n): (λ/K) times a mixture ratio combining immediate deletion-of-insert
    and potential matching to leaf letters, again via inside–outside sums.
- Two APIs:
  - Grouped: aggregates substitution and insertion rates into “match any leaf token value appearing in x1”
    versus “all other tokens,” avoiding O(K) storage.
  - Full tensor: returns full per-token tensors of shape (K, n) for substitutions and (K, n+1) for insertions.

Numerical notes
- All inside–outside computations are carried out in log-space with a stable logaddexp.
- Backward sampling protects against underflow: if all weights are 0 due to underflow, a valid move is chosen
  deterministically to complete a path.

Shapes and symbols (n := length(X_t), m := length(x1), K := p.k)
- Grouped Doob:
  - del :: (n,)
  - sub_match :: (U, n), tokens U that appear at least once in x1
  - sub_other :: (n,), aggregated over the “other” class at each position
  - sub_match_tokens :: (U,), the concrete token IDs for the “match” columns
  - sub_other_count :: (n,), multiplicity for the “other” class per position i
  - ins_match :: (U, n+1)
  - ins_other :: (n+1,)
  - ins_other_count :: Int, multiplicity K - U for insertions
  - hcur :: scalar likelihood
- Full-tensor Doob:
  - sub :: (K, n)
  - del :: (n,)
  - ins :: (K, n+1)
  - hcur :: scalar likelihood
=#


"""
    UniformDiscretePoissonIndelProcess(λ, μ, α, k)

Model parameters for the Uniform Discrete PIP:
- λ::T: insertion rate.
- μ::T: deletion rate.
- α::T: substitution rate parameter; off-diagonal instantaneous rate is α/k.
- k::Int: alphabet size (tokens 1…k).

For a branch of length s:
- Survival of an existing letter: exp(-μ s).
- Substitution kernel:
    Pdiag(s) = exp(-α s) + (1 - exp(-α s))/k
    Poff(s)  = (1 - exp(-α s))/k
- Per-token insertion probability mass:
    Iins(s) = λ * ((1 - exp(-μ s))/μ) / k
"""

struct UniformDiscretePoissonIndelProcess{T}  <: DiscreteIndelProcess
  λ::T          # insertion rate
  μ::T          # deletion rate
  α::T          # substitution "decay" rate; pow(s) = exp(-α s)
  k::Int        # alphabet size (tokens are Ints 1…k)
  transform::Function # maps unconstrained logits to valid rates (default NNlib.softplus)
end

# Backward-compatible convenience constructors
UniformDiscretePoissonIndelProcess(k ; lambda=1.0, mu=1.0, alpha=1.0, transform = NNlib.softplus) =
    UniformDiscretePoissonIndelProcess(lambda, mu, alpha, k, transform)

UniformDiscretePoissonIndelProcess(λ::T, μ::T, α::T, k::Int) where {T} =
    UniformDiscretePoissonIndelProcess{T}(λ, μ, α, k, NNlib.softplus)


prefix(S::DiscreteState, k::Int) = DiscreteState(max(S.K,k), vcat(k, S.state))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
survival(μ,t) = exp(-μ*t)
pow_sub(α,t)  = exp(-α*t)

"""
    rand_other_excl2(rng, k, a, b) -> Int

Sample uniformly from {1,…,k} excluding {a,b}.
Assumes 1 ≤ a,b ≤ k and a ≠ b.
"""
function rand_other_excl2(rng::AbstractRNG, k::Int, a::Int, b::Int)
  x = rand(rng, 1:k-2)
  x >= min(a,b) && (x += 1)
  x >= max(a,b) && (x += 1)
  return x
end


"""
    rand_other_excl1(rng, k, a) -> Int

Sample uniformly from {1,…,k} excluding {a}.
Assumes 1 ≤ a ≤ k.
"""
function rand_other_excl1(rng::AbstractRNG, k::Int, a::Int)
  x = rand(rng, 1:k-1)
  x >= a && (x += 1)
  return x
end

# ─────────────────────────────────────────────────────────────────────────────
# Two-branch kernels specialized to UniformDiscrete
# ─────────────────────────────────────────────────────────────────────────────
"""
    KernelsUD

Container for per-step posterior proportions used by the two-branch alignment DP:
- pA::T: probability of an A-step (x0-only) per letter, aggregated over tokens.
- pB::T: probability of a B-step (x1-only) per letter, aggregated over tokens.
- pR_diag::T: probability of an R-step when x0[i] == x1[j].
- pR_off::T: probability of an R-step when x0[i] != x1[j].
These are normalized so that, at any (i,j), the three out-going transitions sum appropriately.
"""
struct KernelsUD{T}
  pA::T
  pB::T
  pR_diag::T
  pR_off::T
end

"""
    PrecompUD

Precomputed branch- and time-specific constants at time t:
- t, s = t, 1 - t
- s1 = exp(-μ t), s2 = exp(-μ (1 - t))
- pow_t = exp(-α t), pow_s = exp(-α (1 - t))
- P1_diag, P1_off: substitution kernel on left branch (length t)
- P2_diag, P2_off: substitution kernel on right branch (length 1 - t)
- Iins_t, Iins_s: per-token insertion mass on left/right branches
"""
struct PrecompUD{T}
  t::T
  s::T               # 1 - t
  s1::T              # e^{-μ t}
  s2::T              # e^{-μ (1-t)}
  pow_t::T           # e^{-α t}
  pow_s::T           # e^{-α (1-t)}
  P1_diag::T; P1_off::T
  P2_diag::T; P2_off::T
  Iins_t::T          # λ * ((1 - e^{-μ t})/μ) / K
  Iins_s::T          # λ * ((1 - e^{-μ (1-t)})/μ) / K
end

"""
    make_precomp(p, t) -> PrecompUD

Build `PrecompUD` with survival, substitution kernels and per-token insertion masses
for the left (length t) and right (length 1 - t) branches.
"""
function make_precomp(p::UniformDiscretePoissonIndelProcess{T}, t::T) where T
  s  = one(T) - t
  s1 = survival(p.μ, t)
  s2 = survival(p.μ, s)
  powt = pow_sub(p.α, t)
  pows = pow_sub(p.α, s)
  P1_diag = powt + (1 - powt)/p.k
  P1_off  = (1 - powt)/p.k
  P2_diag = pows + (1 - pows)/p.k
  P2_off  = (1 - pows)/p.k
  Iins_t  = p.λ * ((1 - s1)/p.μ) / p.k
  Iins_s  = p.λ * ((1 - s2)/p.μ) / p.k
  PrecompUD(t, s, s1, s2, powt, pows, P1_diag, P1_off, P2_diag, P2_off, Iins_t, Iins_s)
end

"""
    make_kernels(p, C) -> KernelsUD

Build the two-branch event proportions (A, B, R) used by the alignment DP:
- Aggregates over token identities for A/B and splits R into diagonal vs off-diagonal cases.
- Normalizes by total mass Z so the three transitions at a grid cell (i,j) are comparable.
"""
function make_kernels(p::UniformDiscretePoissonIndelProcess{T}, C::PrecompUD{T}) where T
  K = p.k
  # eA,eB per-letter (before normalization)
  eA_per = C.s1*(1 - C.s2)/K + p.λ*((1 - C.s1)/p.μ)/K          # independent of α
  eB_per = C.s2*(1 - C.s1)/K + p.λ*((1 - C.s2)/p.μ)/K
  eA_tot = K*eA_per
  eB_tot = K*eB_per
  # eR for a==b and a!=b (before normalization)
  s12 = C.s1*C.s2
  eR_same = (1/K) * s12 * (C.P1_diag*C.P2_diag + (K-1)*C.P1_off*C.P2_off)
  eR_diff = (1/K) * s12 * (C.P1_diag*C.P2_off  + C.P1_off*C.P2_diag + (K-2)*C.P1_off*C.P2_off)
  sum_eR = K*eR_same + K*(K-1)*eR_diff
  Z = sum_eR + eA_tot + eB_tot
  KernelsUD(eA_per/Z, eB_per/Z, eR_same/Z, eR_diff/Z)
end

# ─────────────────────────────────────────────────────────────────────────────
# Forward DP and backward sample using KernelsUD
# ─────────────────────────────────────────────────────────────────────────────
"""
    Event

Alignment event in the two-branch DP:
- typ::Symbol: one of :A (x0-only), :B (x1-only), :R (paired).
- i::Int, j::Int: 1-based indices into x0 and x1 at the step (0 used for the “other side” of an indel).
"""
struct Event
  typ::Symbol
  i::Int
  j::Int
end


"""
    sample_alignment_ud(rng, A, B, KUD) -> (events::Vector{Event}, h::T)

Forward dynamic program followed by backward sampling of an alignment path between sequences A and B:
- Uses `KUD::KernelsUD` to weight A/B/R steps, with R split by token equality.
- Returns the sampled sequence of `Event`s and the total path mass `h = F[n+1, m+1]`.
"""
function sample_alignment_ud(rng::AbstractRNG,
                             A::Vector{Int}, B::Vector{Int},
                             KUD::KernelsUD{T}) where T
  n, m = length(A), length(B)
  F = zeros(T, n+1, m+1); F[1,1] = one(T)

  # forward DP
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f * KUD.pA
    end
    if j < m
      F[i+1,j+2] += f * KUD.pB
    end
    if i < n && j < m
      pr = (A[i+1] == B[j+1]) ? KUD.pR_diag : KUD.pR_off
      F[i+2,j+2] += f * pr
    end
  end

  # backward sampling with explicit bounds guards
  events = Vector{Event}()
  i, j = n, m
  while i>0 || j>0
    #### Block introduced to fix a rare-ish zero-index bug. Needs more careful thought.
    # Defensive guards against impossible negatives due to rare numeric issues. BM: sus.
    if i < 0; i = 0; end
    if j < 0; j = 0; end
    #### End of Block introduced to fix a zero-index bug. Needs more careful thought.
    wA = zero(T)
    if i > 0
      wA = F[i, j+1] * KUD.pA
    end

    wB = zero(T)
    if j > 0
      wB = F[i+1, j] * KUD.pB
    end

    wR = zero(T)
    if i > 0 && j > 0
      wR = F[i, j] * ((A[i] == B[j]) ? KUD.pR_diag : KUD.pR_off)
    end

    wTot = wA + wB + wR
    # If numerical underflow makes wTot==0, prefer any valid move deterministically
    if !(wTot > 0)
      if i > 0 && j > 0
        push!(events, Event(:R, i, j)); i -= 1; j -= 1
      elseif i > 0
        push!(events, Event(:A, i, 0)); i -= 1
      else
        push!(events, Event(:B, 0, j)); j -= 1
      end
      continue
    end

    u = rand(rng) * float(wTot)
    if u < wA
      push!(events, Event(:A, i, 0));  i -= 1
    elseif u < wA + wB
      push!(events, Event(:B, 0, j));  j -= 1
    else
      push!(events, Event(:R, i, j));  i -= 1;  j -= 1
    end
  end

  reverse!(events)
  return events, F[n+1,m+1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Expand visible + invisible ⇒ X_t (ancestral r sampling in O(1))
# ─────────────────────────────────────────────────────────────────────────────
"""
    sample_Xt_ud(rng, p, x0, x1, C) -> (Xt::Vector{Int}, events::Vector{Event})

Samples an ancestral sequence X_t at time t given endpoints x0 (at 0) and x1 (at 1),
under the Uniform Discrete PIP bridge:
- Builds `KernelsUD` from `p` and `C::PrecompUD` at time t, samples an alignment path,
  then samples ancestral tokens r for each event as:
    - :R: proportional to P1(r→x0[i]) * P2(r→x1[j]).
    - :A or :B: mixture of “present at t” vs “inserted after t” on that branch; if present, r|descendant via P1/P2.
- Finally, adds “ghost” insertions in X_t: Poisson((λ/μ) (1 - s1) (1 - s2)) many, uniform placement and token.

Returns the sampled `Xt` and the alignment `events`.
"""
function sample_Xt_ud(rng::AbstractRNG, p::UniformDiscretePoissonIndelProcess{T},
                      x0::Vector{Int}, x1::Vector{Int}, C::PrecompUD{T}) where T
  KUD = make_kernels(p, C)
  events, _ = sample_alignment_ud(rng, x0, x1, KUD)

  Xt = Int[]
  K = p.k

  for ev in events
    if ev.typ === :R
      #### Block introduced to fix a rare-ish zero-index bug. Needs more careful thought.
      # Rarely, numerical edge-cases can yield an :R with a border index 0.
      # Treat these as single-side events to avoid invalid indexing.
      if ev.i == 0 && ev.j > 0
        # degrade to :B logic
        b = x1[ev.j]
        pre = C.s2*(1 - C.s1) / K
        ins = (p.λ/p.μ) * (1 - C.s2) / K
        if rand(rng) < pre/(pre+ins)
          w_b   = C.P2_diag
          w_oth = C.P2_off
          tot = w_b + (K-1)*w_oth
          if rand(rng) < w_b/tot
            push!(Xt, b)
          else
            push!(Xt, rand_other_excl1(rng, K, b))
          end
        end
        continue
      elseif ev.j == 0 && ev.i > 0
        # degrade to :A logic
        a = x0[ev.i]
        pre = C.s1*(1 - C.s2) / K
        ins = (p.λ/p.μ) * (1 - C.s1) / K
        if rand(rng) < pre/(pre+ins)
          w_a   = C.P1_diag
          w_oth = C.P1_off
          tot = w_a + (K-1)*w_oth
          if rand(rng) < w_a/tot
            push!(Xt, a)
          else
            push!(Xt, rand_other_excl1(rng, K, a))
          end
        end
        continue
      elseif ev.i == 0 && ev.j == 0
        # nothing to do
        continue
      end
      #### End of Block introduced to fix a zero-index bug.
      a = x0[ev.i]; b = x1[ev.j]
      if a == b
        w_a   = C.P1_diag * C.P2_diag
        w_oth = C.P1_off  * C.P2_off
        tot = w_a + (K-1)*w_oth
        if rand(rng) < w_a / tot
          push!(Xt, a)
        else
          push!(Xt, rand_other_excl1(rng, K, a))
        end
      else
        w_a   = C.P1_diag * C.P2_off
        w_b   = C.P1_off  * C.P2_diag
        w_oth = C.P1_off  * C.P2_off
        tot = w_a + w_b + (K-2)*w_oth
        u = rand(rng)*tot
        if u < w_a
          push!(Xt, a)
        elseif u < w_a + w_b
          push!(Xt, b)
        else
          push!(Xt, rand_other_excl2(rng, K, a, b))
        end
      end
    elseif ev.typ === :A
      a = x0[ev.i]
      # present-at-t vs inserted-after-t
      pre = C.s1*(1 - C.s2) / K
      ins = (p.λ/p.μ) * (1 - C.s1) / K
      if rand(rng) < pre/(pre+ins)
        # r=a vs other
        w_a   = C.P1_diag
        w_oth = C.P1_off
        tot = w_a + (K-1)*w_oth
        if rand(rng) < w_a/tot
          push!(Xt, a)
        else
          push!(Xt, rand_other_excl1(rng, K, a))
        end
      end
    else # :B
      b = x1[ev.j]
      pre = C.s2*(1 - C.s1) / K
      ins = (p.λ/p.μ) * (1 - C.s2) / K
      if rand(rng) < pre/(pre+ins)
        w_b   = C.P2_diag
        w_oth = C.P2_off
        tot = w_b + (K-1)*w_oth
        if rand(rng) < w_b/tot
          push!(Xt, b)
        else
          push!(Xt, rand_other_excl1(rng, K, b))
        end
      end
    end
  end

  # ghosts: (λ/μ) * (1 - s1) * (1 - s2); insert uniformly among |Xt|+1 slots
  Λghost = (p.λ/p.μ) * (1 - C.s1) * (1 - C.s2)
  Kghost = rand(Poisson(Λghost))
  for _ in 1:Kghost
    r   = rand(rng, 1:p.k)
    pos = rand(rng, 0:length(Xt))
    insert!(Xt, pos + 1, r)
  end

  return Xt, events
end

#Default RNG
sample_Xt_ud(P, x0, x1, C) = sample_Xt_ud(Random.default_rng(), P, x0, x1, C)

# ─────────────────────────────────────────────────────────────────────────────
# Single-branch DP specialized to UniformDiscrete (no K dependence)
# ─────────────────────────────────────────────────────────────────────────────
"""
    BranchUD

Right-branch (length s = 1 - t) constants for single-branch DP:
- surv::T: e^{-μ s}
- Pdiag, Poff: substitution kernel on the right branch
- Iins::T: per-token insertion mass on the right branch
- q_off::T: instantaneous off-diagonal rate α/K
"""
struct BranchUD{T}
  surv::T          # e^{-μ s}
  Pdiag::T         # pow_s + (1 - pow_s)/K
  Poff::T          # (1 - pow_s)/K
  Iins::T          # λ * ((1 - surv)/μ) / K
  q_off::T         # instantaneous substitution a→b, b≠a: α / K
end

"""
    make_branch_ud(p, C) -> BranchUD

Build right-branch constants from `UniformDiscretePoissonIndelProcess` and `PrecompUD` at time t.
"""
function make_branch_ud(p::UniformDiscretePoissonIndelProcess{T}, C::PrecompUD{T}) where T
  q_off = p.α / p.k
  BranchUD{T}(C.s2, C.P2_diag, C.P2_off, C.Iins_s, q_off)
end

"""
    branch_prob_ud!(F, root, leaf, B) -> h::T

Forward DP for P(leaf | root) on the right branch with uniform-discrete substitutions:
- root, leaf :: Vector{Int}
- F :: preallocated (n+1)×(m+1) matrix; overwritten
- B :: `BranchUD`
Returns the total probability h.
"""
function branch_prob_ud!(F::AbstractMatrix{T}, root::Vector{Int},
                         leaf::Vector{Int}, B::BranchUD{T}) where T
  n, m = length(root), length(leaf)
  F[1:n+1, 1:m+1] .= 0
  F[1,1] = one(T)
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f*(1 - B.surv)
    end
    if j < m
      F[i+1,j+2] += f*B.Iins
    end
    if i < n && j < m
      P = (root[i+1] == leaf[j+1]) ? B.Pdiag : B.Poff
      F[i+2,j+2] += f*B.surv*P
    end
  end
  return F[n+1,m+1]
end

"""
    branch_prob_ud_delete!(F, root, delpos, leaf, B) -> h::T

Same as `branch_prob_ud!` but treats `root` with element at 1-based `delpos` removed
when aligning to `leaf`. Useful for marginalizing deletion-at-position proposals.
"""
function branch_prob_ud_delete!(F::AbstractMatrix{T}, root::Vector{Int}, delpos::Int,
                                leaf::Vector{Int}, B::BranchUD{T}) where T
  n0, m = length(root), length(leaf)
  n = n0 - 1
  F[1:n+1, 1:m+1] .= 0
  F[1,1] = one(T)
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f*(1 - B.surv)
    end
    if j < m
      F[i+1,j+2] += f*B.Iins
    end
    if i < n && j < m
      idx = i + 1
      orig = (idx < delpos) ? idx : idx + 1
      P = (root[orig] == leaf[j+1]) ? B.Pdiag : B.Poff
      F[i+2,j+2] += f*B.surv*P
    end
  end
  return F[n+1,m+1]
end

"""
    branch_prob_ud_insert!(F, root, inspos, c, leaf, B) -> h::T

Same as `branch_prob_ud!` but treats `root` with a symbol `c` inserted after position `inspos`
(0-based gap index; i.e., `inspos=0` inserts before the first symbol, `inspos=n` after the last).
"""
function branch_prob_ud_insert!(F::AbstractMatrix{T}, root::Vector{Int}, inspos::Int, c::Int,
                                leaf::Vector{Int}, B::BranchUD{T}) where T
  n0, m = length(root), length(leaf)
  n = n0 + 1
  F[1:n+1, 1:m+1] .= 0
  F[1,1] = one(T)
  for i in 0:n, j in 0:m
    f = F[i+1,j+1]
    if i < n
      F[i+2,j+1] += f*(1 - B.surv)
    end
    if j < m
      F[i+1,j+2] += f*B.Iins
    end
    if i < n && j < m
      idx = i + 1
      a = (idx == inspos + 1) ? c : ((idx <= inspos) ? root[idx] : root[idx-1])
      P = (a == leaf[j+1]) ? B.Pdiag : B.Poff
      F[i+2,j+2] += f*B.surv*P
    end
  end
  return F[n+1,m+1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Doob rates grouped by match-vs-other classes (no O(K))
# ─────────────────────────────────────────────────────────────────────────────
"""
    doob_ud_grouped_fast(p, Xt, x1, t)

Grouped Doob h-transform rates for Uniform Discrete on the right branch (length 1 - t).
Inputs:
- p::UniformDiscretePoissonIndelProcess, Xt::Vector{Int} (current ancestral sequence at time t), x1::Vector{Int} (leaf), t::T

Returns NamedTuple:
- hcur::T: current likelihood h(Xt → x1 | branch)
- del::Vector{T}               (n,)            deletion rates at each position i
- sub_match::Matrix{T}         (U, n)          substitution rates to tokens that appear in x1
- sub_other::Vector{T}         (n,)            substitution rate aggregated over “other” tokens
- sub_match_tokens::Vector{Int}(U,)            token IDs for the rows of `sub_match`
- sub_other_count::Vector{Int} (n,)            multiplicity of “other” class at each i
- ins_match::Matrix{T}         (U, n+1)        insertion rates for tokens in x1 at each gap
- ins_other::Vector{T}         (n+1,)          insertion rate aggregated over “other” tokens
- ins_other_count::Int                         multiplicity of “other” insertion class (K - U)

All rates are instantaneous hazards based on inside–outside ratios in log-space.
"""
function doob_ud_grouped_fast(p::UniformDiscretePoissonIndelProcess{T}, Xt::Vector{Int}, x1::Vector{Int}, t::T) where T

    local function logaddexp(a::T, b::T) where T
        if a == -Inf; return b; end
        if b == -Inf; return a; end
        if b > a; a, b = b, a; end
        return a + log1p(exp(b - a))
    end

    C = make_precomp(p, t)
    n, m, K = length(Xt), length(x1), p.k

    # Precompute logs of constants
    log_s2  = log(C.s2)
    log_del = log1p(-C.s2)                  # log(1 - s2)
    log_ins = log(p.λ) + log1p(-C.s2) - log(p.μ) - log(K)

    # Forward DP (log)
    logF = fill(-Inf, n+1, m+1); logF[1,1] = zero(T)
    for i in 0:n, j in 0:m
        f = logF[i+1, j+1]; f == -Inf && continue
        if i < n
            logF[i+2, j+1] = logaddexp(logF[i+2, j+1], f + log_del)
        end
        if j < m
            logF[i+1, j+2] = logaddexp(logF[i+1, j+2], f + log_ins)
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            logF[i+2, j+2] = logaddexp(logF[i+2, j+2], f + log_s2 + log(P))
        end
    end
    loghcur = logF[n+1, m+1]

    # Backward DP (log)
    logB = fill(-Inf, n+1, m+1); logB[n+1, m+1] = zero(T)
    for i in n:-1:0, j in m:-1:0
        (i == n && j == m) && continue
        acc = -Inf
        if i < n
            acc = logaddexp(acc, log_del + logB[i+2, j+1])
        end
        if j < m
            acc = logaddexp(acc, log_ins + logB[i+1, j+2])
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            acc = logaddexp(acc, log_s2 + log(P) + logB[i+2, j+2])
        end
        logB[i+1, j+1] = acc
    end

    # Token set in x1
    seen = Dict{Int,Int}(); toks = Int[]
    for b in x1
        haskey(seen,b) || (seen[b] = length(seen)+1; push!(toks,b))
    end
    U = length(toks)

    # Substitutions (inside–outside ratios)
    ΔP = C.P2_diag - C.P2_off
    q_off = p.α / K

    sub_match = zeros(T, U, n)
    sub_other = zeros(T, n)
    sub_other_count = Vector{Int}(undef, n)

    for i in 1:n
        # per-token accumulator over leaf positions, in log-space
        tmp = fill(-Inf, U)
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[i, j] + log_s2 + logB[i+1, j+1]
                tmp[idx] = logaddexp(tmp[idx], val)
            end
        end
        # convert to ratios
        r_tok = similar(tmp)
        for u in 1:U
            r_tok[u] = tmp[u] == -Inf ? zero(T) : exp(tmp[u] - loghcur)
        end
        idx_a = get(seen, Xt[i], 0)
        r_a = (idx_a == 0 || tmp[idx_a] == -Inf) ? zero(T) : exp(tmp[idx_a] - loghcur)

        for u in 1:U
            ratio = max(zero(T), 1 + ΔP * (r_tok[u] - r_a))
            sub_match[u, i] = q_off * ratio
        end
        other_mult = K - 1 - (haskey(seen, Xt[i]) ? U-1 : U)
        sub_other_count[i] = max(other_mult, 0)
        base_ratio = max(zero(T), 1 - ΔP * r_a)
        sub_other[i] = q_off * base_ratio * sub_other_count[i]
    end

    # Deletions (ratios)
    del = zeros(T, n)
    for i in 1:n
        acc = -Inf
        for j in 0:m
            #acc = logaddexp(acc, logF[i, j+1] + log_del + logB[i+1, j+1])
            acc = logaddexp(acc, logF[i, j+1] + logB[i+1, j+1])
        end
        r = acc == -Inf ? zero(T) : exp(acc - loghcur)
        del[i] = p.μ * r
    end

    # Insertions (grouped): fix indexing for match branch: F[s+1,j] * B[s+1,j+1]
    D_ratio  = zeros(T, n+1)      # deletion-of-insert branch ratio
    M_ratio  = zeros(T, U, n+1)   # match-of-insert per token ratio
    for s in 0:n
        # deletion branch sum_j F[s,j] * B[s,j]
        accD = -Inf
        for j in 0:m
            accD = logaddexp(accD, logF[s+1, j+1] + logB[s+1, j+1])
        end
        D_ratio[s+1] = accD == -Inf ? zero(T) : exp(accD - loghcur)

        # match branch sum over j with correct indexing (no out-of-bounds at s=n)
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[s+1, j] + logB[s+1, j+1]   # note: B[s+1, j+1], not s+2
                M_ratio[idx, s+1] += exp(val - loghcur)
            end
        end
    end
    S_ratio = vec(sum(M_ratio; dims=1))  # (n+1,)

    ins_match = zeros(T, U, n+1)
    ins_other = zeros(T, n+1)
    ins_other_count = K - U
    q_ins = p.λ / K #Flag
    
    for s in 0:n
        base_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * S_ratio[s+1] * C.P2_off
        base_ratio = max(zero(T), base_ratio)
        ins_other[s+1] = q_ins * base_ratio * max(ins_other_count, 0)

        for (u, tok) in enumerate(toks)
            r_m = M_ratio[u, s+1]
            val_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * (r_m * C.P2_diag + (S_ratio[s+1] - r_m) * C.P2_off)
            ins_match[u, s+1] = q_ins * max(zero(T), val_ratio)
        end
    end

    return (hcur = exp(loghcur),
            del = del,
            sub_match = sub_match,
            sub_other = sub_other,
            sub_match_tokens = toks,
            sub_other_count = sub_other_count,
            ins_match = ins_match,
            ins_other = ins_other,
            ins_other_count = ins_other_count)
end


"""
    doob_ud_full_tensors_fast(p, Xt, x1, t)

Full-tensor Doob h-transform rates for Uniform Discrete on the right branch (length 1 - t).
Inputs:
- p::UniformDiscretePoissonIndelProcess, Xt::Vector{Int}, x1::Vector{Int}, t::T

Returns NamedTuple:
- sub::Array{T,2}  (K, n): substitution hazards to each token at each position (self-substitution zeroed)
- del::Vector{T}   (n,):   deletion hazards per position
- ins::Array{T,2}  (K, n+1): insertion hazards to each token per gap
- hcur::T: current likelihood

Internally uses log-space forward-backward and token-first storage for cache efficiency.
"""
function doob_ud_full_tensors_fast(p::UniformDiscretePoissonIndelProcess{T}, Xt::Vector{Int}, x1::Vector{Int}, t::T) where T
    # log-space inside–outside and full tensors with token-first layout:
    # Returns:
    #   sub :: (K, n), del :: (n,), ins :: (K, n+1), hcur
    local function logaddexp(a::T, b::T) where T
        if a == -Inf; return b; end
        if b == -Inf; return a; end
        if b > a; a, b = b, a; end
        return a + log1p(exp(b - a))
    end

    C = make_precomp(p, t)
    n, m, K = length(Xt), length(x1), p.k

    log_s2  = log(C.s2)
    log_del = log1p(-C.s2)
    log_ins = log(p.λ) + log1p(-C.s2) - log(p.μ) - log(K)

    # Forward DP (log)
    logF = fill(-Inf, n+1, m+1); logF[1,1] = zero(T)
    for i in 0:n, j in 0:m
        f = logF[i+1, j+1]; f == -Inf && continue
        if i < n
            logF[i+2, j+1] = logaddexp(logF[i+2, j+1], f + log_del)
        end
        if j < m
            logF[i+1, j+2] = logaddexp(logF[i+1, j+2], f + log_ins)
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            logF[i+2, j+2] = logaddexp(logF[i+2, j+2], f + log_s2 + log(P))
        end
    end
    loghcur = logF[n+1, m+1]

    # Backward DP (log)
    logB = fill(-Inf, n+1, m+1); logB[n+1, m+1] = zero(T)
    for i in n:-1:0, j in m:-1:0
        (i == n && j == m) && continue
        acc = -Inf
        if i < n
            acc = logaddexp(acc, log_del + logB[i+2, j+1])
        end
        if j < m
            acc = logaddexp(acc, log_ins + logB[i+1, j+2])
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? C.P2_diag : C.P2_off
            acc = logaddexp(acc, log_s2 + log(P) + logB[i+2, j+2])
        end
        logB[i+1, j+1] = acc
    end

    # Token set in x1
    seen = Dict{Int,Int}(); toks = Int[]
    for b in x1
        haskey(seen,b) || (seen[b] = length(seen)+1; push!(toks,b))
    end
    U = length(toks)

    ΔP   = C.P2_diag - C.P2_off
    q_off = p.α / K
    q_ins = p.λ / K #Flag
    
    # Substitutions: token-first (K, n)
    sub = zeros(T, K, n)
    for i in 1:n
        # per-token accumulator over leaf positions, in log-space
        tmp = fill(-Inf, U)
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[i, j] + log_s2 + logB[i+1, j+1]
                tmp[idx] = logaddexp(tmp[idx], val)
            end
        end
        # convert to ratios
        r_tok = similar(tmp)
        for u in 1:U
            r_tok[u] = tmp[u] == -Inf ? zero(T) : exp(tmp[u] - loghcur)
        end
        idx_a = get(seen, Xt[i], 0)
        r_a = (idx_a == 0 || tmp[idx_a] == -Inf) ? zero(T) : exp(tmp[idx_a] - loghcur)

        # base fill (other tokens, including those not in x1)
        base_ratio = max(zero(T), 1 - ΔP * r_a)
        fill!(view(sub, :, i), q_off * base_ratio)
        # overwrite specific tokens in x1
        for (u, tok) in enumerate(toks)
            ratio = max(zero(T), 1 + ΔP * (r_tok[u] - r_a))
            sub[tok, i] = q_off * ratio
        end
        # no self-substitution
        sub[Xt[i], i] = zero(T)
    end

    # Deletions: (n,)
    del = zeros(T, n)
    for i in 1:n
        acc = -Inf
        for j in 0:m
            #acc = logaddexp(acc, logF[i, j+1] + log_del + logB[i+1, j+1])
            acc = logaddexp(acc, logF[i, j+1] + logB[i+1, j+1])
        end
        r = acc == -Inf ? zero(T) : exp(acc - loghcur)
        del[i] = p.μ * r
    end

    # Insertions: token-first (K, n+1); fix match indexing to B[s+1, j+1]
    D_ratio = zeros(T, n+1)           # deletion-of-insert ratio
    M_ratio = zeros(T, U, n+1)        # match-of-insert per token ratio
    for s in 0:n
        # deletion branch sum_j F[s,j] * B[s,j]
        accD = -Inf
        for j in 0:m
            accD = logaddexp(accD, logF[s+1, j+1] + logB[s+1, j+1])
        end
        D_ratio[s+1] = accD == -Inf ? zero(T) : exp(accD - loghcur)
        # match branch: sum over j with correct indexing
        for j in 1:m
            idx = get(seen, x1[j], 0)
            if idx != 0
                val = logF[s+1, j] + logB[s+1, j+1]   # note: B[s+1, j+1]
                M_ratio[idx, s+1] += exp(val - loghcur)
            end
        end
    end
    S_ratio = vec(sum(M_ratio; dims=1))  # (n+1,)

    ins = zeros(T, K, n+1)
    for s in 0:n
        base_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * S_ratio[s+1] * C.P2_off
        base_ratio = max(zero(T), base_ratio)
        fill!(view(ins, :, s+1), q_ins * base_ratio)
        for (u, tok) in enumerate(toks)
            r_m = M_ratio[u, s+1]
            val_ratio = (1 - C.s2) * D_ratio[s+1] + C.s2 * (r_m * C.P2_diag + (S_ratio[s+1] - r_m) * C.P2_off)
            ins[tok, s+1] = q_ins * max(zero(T), val_ratio)
        end
    end

    return (sub = sub, del = del, ins = ins, hcur = exp(loghcur))
end


function doob_ud_full_tensors_aligned(p::UniformDiscretePoissonIndelProcess{T},
                                      Xt::Vector{Int}, x1::Vector{Int}, t::T,
                                      events::Vector{Event}) where {T}
    # Right-branch constants at time t
    C = make_precomp(p, t)
    s2    = C.s2
    Pdiag = C.P2_diag
    Poff  = C.P2_off
    K     = p.k
    n     = length(Xt)
    m     = length(x1)

    # instantaneous rates
    q_off = p.α / K        # off-diagonal substitution per token
    q_ins = p.λ / K        # per-token insertion rate

    # ── Decode alignment into obligations (match/delete per site, B-queues per gap)
    tag        = Vector{Symbol}(undef, n)   # :match or :delete
    match_tok  = zeros(Int, n)              # valid iff :match
    gap_queues = [Int[] for _ in 1:(n+1)]   # store LEAF TOKENS per gap (0..n)

    ic = 0; jc = 0; gap = 0
    for ev in events
        if ev.typ === :R
            ic += 1; jc += 1
            tag[ic] = :match
            match_tok[ic] = x1[jc]
            gap = ic
        elseif ev.typ === :A
            ic += 1
            tag[ic] = :delete
            gap = ic
        elseif ev.typ === :B
            jc += 1
            # push LEAF TOKEN (x1[jc]) into this gap’s queue
            Base.push!(gap_queues[gap + 1], x1[jc])
        else
            error("Unknown event typ=$(ev.typ)")
        end
    end
    @assert ic == n "alignment does not consume all Xt positions (got $ic, need $n)"
    @assert jc == m "alignment does not consume all x1 positions (got $jc, need $m)"

    # ── Allocate hazards
    sub = zeros(T, K, n)
    del = zeros(T, n)
    ins = zeros(T, K, n+1)

    # ── Per-site hazards
    @inbounds for i in 1:n
        a = Xt[i]
        if tag[i] === :match
            b = match_tok[i]
            denom = (a == b) ? Pdiag : Poff
            for c in 1:K
                if c != a
                    num = (c == b) ? Pdiag : Poff
                    sub[c, i] = q_off * (num / denom)
                end
            end
            del[i] = zero(T)                    # cannot delete a must-match site
        else
            # must-delete: substitutions are neutral wrt h ⇒ plain q_off
            for c in 1:K
                if c != a
                    sub[c, i] = q_off
                end
            end
            # strict Doob deletion scaling μ/(1 - s2)
            del[i] = p.μ / max(eps(T), (one(T) - s2))
        end
    end

    # ── Per-gap insertions (FIXED: include (L+1) transient multiplicity)
    # base (transient) = q_ins * (1 - s2) * (L + 1)
    # productive term  = (μ * s2 / (1 - s2)) * Σ_{r in queue} P(c→r)
    # with Σ P(c→r) = cnt[c]*Pdiag + (L - cnt[c])*Poff
    @inbounds for s in 0:n
        q = gap_queues[s+1]     # Vector{Int} of leaf TOKENS in this gap
        L = length(q)

        # delete-before-leaf in any of (L+1) slots along the fixed path
        base = q_ins * (one(T) - s2) * (L + 1)

        if L == 0
            for c in 1:K
                ins[c, s+1] = base
            end
        else
            cnt = zeros(Int, K)
            for r in q
                cnt[r] += 1
            end
            factor = p.μ * s2 / max(eps(T), (one(T) - s2))
            for c in 1:K
                sumP = cnt[c] * Pdiag + (L - cnt[c]) * Poff
                ins[c, s+1] = base + factor * sumP
            end
        end
    end

    return (sub = sub, del = del, ins = ins)
end




function bridge(p::UniformDiscretePoissonIndelProcess, x0::DiscreteState{<:AbstractArray{<:Signed}}, x1::DiscreteState{<:AbstractArray{<:Signed}}, t)
    if ndims(x0.state) != 1
        error("bridge for UniformDiscretePoissonIndelProcess only implemented for 1D DiscreteState")
    end
    C = make_precomp(p, t)
    flat_xt, _ = sample_Xt_ud(p, tensor(x0), tensor(x1), C)
    xt = DiscreteState(x0.K, flat_xt)
    return xt
end


# # ─────────────────────────────────────────────────────────────────────────────
# # PIP Doob-matching loss (positive Bregman on rates)
# # Needs more thought
# # ─────────────────────────────────────────────────────────────────────────────
# function _pos_breg(p::AbstractArray{T}, q::AbstractArray{T}; eps = T(1e-8)) where {T}
#   return p .* (log.(p .+ eps) .- log.(q .+ eps)) .- p .+ q
# end

# ─────────────────────────────────────────────────────────────────────────────
# Bregman losses on nonnegative rates (per-entry, elementwise).
# p = teacher/target rates, q = model rates
# Return same shape as inputs; reduce with scaledmaskedmean(...).
# ─────────────────────────────────────────────────────────────────────────────

# (A) Generalized KL / Poisson deviance (you already have as `_pos_breg`)
@inline function _pos_breg_gkl(p::AbstractArray{T}, q::AbstractArray{T}; eps::T=T(1e-8)) where {T}
  p̃ = max.(p, eps);  q̃ = max.(q, eps)
  return p̃ .* (log.(p̃) .- log.(q̃)) .- p̃ .+ q̃
end

# (B) Quadratic (MSE): φ(r)=½‖r‖² ⇒ Dφ(p,q)=½‖p−q‖²
@inline function _pos_breg_mse(p::AbstractArray{T}, q::AbstractArray{T}) where {T}
  return T(0.5) .* (p .- q).^2
end

# (C) Reverse generalized KL: D_rev(p,q) = D_gKL(q || p)
@inline function _pos_breg_revkl(p::AbstractArray{T}, q::AbstractArray{T}; eps::T=T(1e-8)) where {T}
  p̃ = max.(p, eps);  q̃ = max.(q, eps)
  return q̃ .* (log.(q̃) .- log.(p̃)) .- q̃ .+ p̃
end

# (D) Itakura–Saito (scale-invariant): φ(r)=−∑ log r
@inline function _pos_breg_is(p::AbstractArray{T}, q::AbstractArray{T}; eps::T=T(1e-8)) where {T}
  p̃ = max.(p, eps);  q̃ = max.(q, eps)
  return (p̃ ./ q̃) .- log.(p̃ ./ q̃) .- one(T)
end

# (E) β-divergence family (unifies MSE, gKL, IS)
# Limits: β→2 ⇒ MSE, β→1 ⇒ gKL, β→0 ⇒ IS
@inline function _pos_breg_beta(p::AbstractArray{T}, q::AbstractArray{T}, β::Real; eps::T=T(1e-8)) where {T}
  if abs(β - 2) < 1e-12
      return _pos_breg_mse(p, q)
  elseif abs(β - 1) < 1e-12
      return _pos_breg_gkl(p, q; eps=eps)
  elseif abs(β) < 1e-12
      return _pos_breg_is(p, q; eps=eps)
  else
      p̃ = max.(p, eps);  q̃ = max.(q, eps)
      βT  = T(β)
      c   = one(T) / (βT * (βT - one(T)))
      return c .* (p̃ .^ βT .+ (βT - one(T)) .* (q̃ .^ βT) .- βT .* (p̃ .* (q̃ .^ (βT - one(T)))))
  end
end

# (F) Weighted/Mahalanobis MSE: φ(r)=½ rᵀW r ⇒ Dφ=½ (p−q)ᵀW(p−q)
# Pass per-entry weights w ≥ 0 (diagonal W). Note: strictly a Bregman if w is fixed.
@inline function _pos_breg_wmse(p::AbstractArray{T}, q::AbstractArray{T}, w::AbstractArray{T}) where {T}
  return T(0.5) .* w .* (p .- q).^2
end

# (G) Pseudo-Huber Bregman (smooth robust)
# φ_δ(x)=δ²( √(1+(x/δ)²) − 1 ),  ∇φ_δ(s)= s / √(1+(s/δ)²)
@inline function _pos_breg_phuber(p::AbstractArray{T}, q::AbstractArray{T}; delta::T=T(1.0), eps::T=T(1e-12)) where {T}
  δ  = max(delta, eps)
  invδ = one(T) / δ
  s_p = sqrt.(one(T) .+ (p .* invδ).^2)
  s_q = sqrt.(one(T) .+ (q .* invδ).^2)
  ϕp  = (δ*δ) .* (s_p .- one(T))
  ϕq  = (δ*δ) .* (s_q .- one(T))
  gradϕq = q ./ s_q
  return ϕp .- ϕq .- gradϕq .* (p .- q)
end

# Choose the Bregman at call-time (dtype-safe: casts eps/delta to eltype(p) on use)
@inline function select_breg(; kind::Symbol=:beta, β::Real=1.0, eps=1e-8, delta=0.1)
  if kind === :gkl
    return (p,q) -> _pos_breg_gkl(p, q; eps = convert(eltype(p), eps))
  elseif kind === :mse
    return _pos_breg_mse
  elseif kind === :is
    return (p,q) -> _pos_breg_is(p, q; eps = convert(eltype(p), eps))
  elseif kind === :revkl
    return (p,q) -> _pos_breg_revkl(p, q; eps = convert(eltype(p), eps))
  elseif kind === :beta
    return (p,q) -> _pos_breg_beta(p, q, β; eps = convert(eltype(p), eps))
  elseif kind === :phuber
    return (p,q) -> _pos_breg_phuber(p, q; delta = convert(eltype(p), delta),
                                         eps   = convert(eltype(p), eps))
  else
    error("Unknown loss kind: $kind")
  end
end

# Example usage in your floss:
function floss(P::fbu(UniformDiscretePoissonIndelProcess), Xt::MaskedState{<:DiscreteState}, X̂₁, G::Guide, c;
             loss_kind::Symbol=:beta, β::Real=1.0, eps=1e-8, delta=0.1)

  sub_rates = P.transform(X̂₁.sub)
  del_rates = P.transform(X̂₁.del)
  ins_rates = P.transform(X̂₁.ins)

  ohXt = tensor(onehotbatch(clamp.(tensor(Xt)[2:end,:], 1, P.k), 1:P.k))
  sub_rates = sub_rates .* (1 .- ohXt)

  breg = select_breg(; kind=loss_kind, β=β, eps=eps, delta=delta)

  return  scaledmaskedmean(breg(G.H.sub, sub_rates), c, getlmask(G)) +
          scaledmaskedmean(breg(G.H.del, del_rates), c, getlmask(G)) +
          scaledmaskedmean(breg(G.H.ins, ins_rates), c, getlmask(Xt))
end


# #Xt must be a MaskedState with the prefix token
# function floss(P::fbu(UniformDiscretePoissonIndelProcess), Xt::MaskedState{<:DiscreteState}, X̂₁, G::Guide, c; loss = _pos_breg)
#   sub_rates = P.transform(X̂₁.sub)
#   del_rates = P.transform(X̂₁.del)
#   ins_rates = P.transform(X̂₁.ins)
#   # Zero self-substitutions using current tokens in Xt
#   #Ick. This is because of the mismatch between the model's "K" and the process's "k", where "batch" gives elements in the range of the former for padded tokens
#   ohXt = tensor(onehotbatch(clamp.(tensor(Xt)[2:end,:], 1, P.k), 1:P.k)) 
#   sub_rates = sub_rates .* (1 .- ohXt)
#   # Positive Bregman D(tgt || pred)
#   loss_val = scaledmaskedmean(loss(G.H.sub, sub_rates), c, getlmask(G)) +
#              scaledmaskedmean(loss(G.H.del, del_rates), c, getlmask(G)) +
#              scaledmaskedmean(loss(G.H.ins, ins_rates), c, getlmask(Xt)) #Use the Xt mask, which has a prefix, for the subs!
#   return loss_val
# end

#=
_like(like, ::Type{T}, dims...) where {T} = similar(like, T, dims...)

function _pos_breg(p::AbstractArray{T}, q::AbstractArray{T}; eps = T(1e-8)) where {T}
  # Elementwise D(p || q) for Poisson-like positive Bregman:
  # p * (log p - log q) - p + q
  return p .* (log.(p .+ eps) .- log.(q .+ eps)) .- p .+ q
end

"""
Xt must be a MaskedState with the prefix token.
This version mirrors one-hot and masks to the same device as `sub_rates` (so CPU/GPU works transparently).
"""
function floss(P::fbu(UniformDiscretePoissonIndelProcess),
               Xt::MaskedState{<:DiscreteState},
               X̂₁, G::Guide, c)

  # Predicted rates (they may already be on CPU or GPU depending on caller)
  sub_rates = P.transform(X̂₁.sub)
  del_rates = P.transform(X̂₁.del)
  ins_rates = P.transform(X̂₁.ins)

  # Mirror Guide payload (target rates) to the same device as `sub_rates`
  H = G.H
  Hsub = (eltype(H.sub) === eltype(sub_rates)) && (typeof(H.sub) === typeof(sub_rates)) ?
         H.sub : (_tmp = _like(sub_rates, eltype(H.sub), size(H.sub)...); copyto!(_tmp, H.sub); _tmp)
  Hdel = (eltype(H.del) === eltype(del_rates)) && (typeof(H.del) === typeof(del_rates)) ?
         H.del : (_tmp = _like(del_rates, eltype(H.del), size(H.del)...); copyto!(_tmp, H.del); _tmp)
  Hins = (eltype(H.ins) === eltype(ins_rates)) && (typeof(H.ins) === typeof(ins_rates)) ?
         H.ins : (_tmp = _like(ins_rates, eltype(H.ins), size(H.ins)...); copyto!(_tmp, H.ins); _tmp)

  # Build self-substitution one-hot from Xt (always CPU ints), then mirror to device of sub_rates
  K = P.k
  xt_tokens = clamp.(tensor(Xt)[2:end, :], 1, K)  # drop prefix, ensure 1..K
  oh_host = tensor(onehotbatch(xt_tokens, 1:K))   # CPU dense
  oh_dev  = (typeof(oh_host) === typeof(sub_rates)) && (eltype(oh_host) === eltype(sub_rates)) ?
            oh_host : (_tmp = _like(sub_rates, eltype(oh_host), size(oh_host)...); copyto!(_tmp, oh_host); _tmp)

  # Zero self-substitutions
  sub_rates = sub_rates .* (1 .- oh_dev)

  # Mirror masks to device (as Bool)
  lmaskG_host = getlmask(G)
  lmaskX_host = getlmask(Xt)
  lmaskG_dev  = (typeof(lmaskG_host) === typeof(sub_rates)) && (eltype(lmaskG_host) === eltype(sub_rates)) ?
                 lmaskG_host :
                 (_tmp = _like(sub_rates, Bool, size(lmaskG_host)...); copyto!(_tmp, lmaskG_host); _tmp)
  lmaskX_dev  = (typeof(lmaskX_host) === typeof(sub_rates)) && (eltype(lmaskX_host) === eltype(sub_rates)) ?
                 lmaskX_host :
                 (_tmp = _like(sub_rates, Bool, size(lmaskX_host)...); copyto!(_tmp, lmaskX_host); _tmp)

  # Mirror c to device if it’s an array
  c_dev = c
  if c isa AbstractArray
      c_dev = (typeof(c) === typeof(sub_rates)) && (eltype(c) === eltype(sub_rates)) ?
              c : (_tmp = _like(sub_rates, eltype(c), size(c)...); copyto!(_tmp, c); _tmp)
  end

  # Positive Bregman D(tgt || pred) with masks on the same device as the data
  loss = scaledmaskedmean(_pos_breg(Hsub, sub_rates), c_dev, lmaskG_dev) +
         scaledmaskedmean(_pos_breg(Hdel, del_rates), c_dev, lmaskG_dev) +
         scaledmaskedmean(_pos_breg(Hins, ins_rates), c_dev, lmaskX_dev)  # Use Xt mask for insertions

  return loss
end

=#
# ─────────────────────────────────────────────────────────────────────────────
# Fused front-end: sample + grouped Doob
# ─────────────────────────────────────────────────────────────────────────────
#=
"""
    sample_and_doob_ud(rng, p, x0, x1, t) -> (Xt, events, rates)

Convenience wrapper:
1) Build precomputation at time t, 2) sample `Xt` and alignment events,
3) compute Doob rates (currently full-tensor).
"""

function sample_and_doob_ud(rng::AbstractRNG, p::UniformDiscretePoissonIndelProcess{T},
                            x0::Vector{Int}, x1::Vector{Int}, t::T) where T
  @time C = make_precomp(p, t)
  @time Xt, events = sample_Xt_ud(rng, p, x0, x1, C)
  #@time rates = doob_ud_grouped_fast(p, Xt, x1, t)
  @time rates = doob_ud_full_tensors_fast(p, Xt, x1, t)
  return Xt, events, rates
end
=#


#=
using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../")
using Flowfusion, ForwardBackward, StatsBase
P = UniformDiscretePoissonIndelProcess(10, lambda = 0.5, mu = 0.5, alpha = 0.5)
P = UniformDiscretePoissonIndelProcess(10, lambda = 0.1, mu = 0.1, alpha = 0.0)
x0 = DiscreteState(20, [1,2,3])
x1 = DiscreteState(20, [1,4,3])
t = 0.001
bridges = [tensor(bridge(P, x0, x1, t)) for _ in 1:1000];
sort(collect(countmap(bridges)), by = last, rev=true)
t = 0.5
bridges = [tensor(bridge(P, x0, x1, t)) for _ in 1:1000];
sort(collect(countmap(bridges)), by = last, rev=true)
t = 0.999
bridges = [tensor(bridge(P, x0, x1, t)) for _ in 1:1000];
sort(collect(countmap(bridges)), by = last, rev=true)



P = UniformDiscretePoissonIndelProcess(10, lambda = 0.1, mu = 0.1, alpha = 0.1)
x0 = DiscreteState(20, [1,4])
x1 = DiscreteState(20, [1])
xt = bridge(P, x0, x1, 0.5)


P = UniformDiscretePoissonIndelProcess(10, lambda = 0.1, mu = 0.1, alpha = 0.1); #Called via convenience constructor
xt = [1,4];
x1 = [1];
rates = Flowfusion.doob_ud_full_tensors_fast(P, xt, x1, 0.999);
rates.del



K = 6
p = UniformDiscretePoissonIndelProcess(0.01, 0.01, 0.01, K)
x0 = DiscreteState(K, [1,4])
x1 = DiscreteState(K, [1,2,4])
times = [0.2, 0.5, 0.8]

# bridge marginals
Nbridge = 10_000
bridge_counts = Dict{Float64, Dict{Vector{Int}, Int}}()
for t in times
    bridge_counts[t] = Dict{Vector{Int}, Int}()
    C = Flowfusion.make_precomp(p, t)
    for _ in 1:Nbridge
        xt, _ = Flowfusion.sample_Xt_ud(p, tensor(x0), tensor(x1), C)
        get!(bridge_counts[t], xt, 0)
        bridge_counts[t][xt] += 1
    end
end

bridge_counts[0.2]

# step marginals via small steps, recording at times
Nstep = 10_000
dt = 0.01
grid = collect(0.0:dt:1.0)
function record!(d::Dict{Vector{Int}, Int}, x::Vector{Int})
    get!(d, x, 0); d[x] += 1
end
step_counts = Dict(t => Dict{Vector{Int}, Int}() for t in times)
for _ in 1:Nstep
    Xt = x0
    for k in 1:(length(grid)-1)
        s1, s2 = grid[k], grid[k+1]
        rates = Flowfusion.doob_ud_full_tensors_fast(p, tensor(Xt), tensor(x1), s1)
        guide = Flowfusion.Guide((sub=rates.sub, del=rates.del, ins=rates.ins))
        Xt = Flowfusion.step(p, Xt, guide, s1, s2)
        if any(abs.(s2 .- times) .< 1e-8)
            record!(step_counts[s2], tensor(Xt))
        end
    end
end

step_counts[0.2]

# compare via total variation distance
function tvd(d1::Dict{Vector{Int}, Int}, n1::Int, d2::Dict{Vector{Int}, Int}, n2::Int)
    keys_all = union(keys(d1), keys(d2))
    s = 0.0
    for k in keys_all
        p = get(d1, k, 0) / n1
        q = get(d2, k, 0) / n2
        s += abs(p - q)
    end
    return 0.5 * s
end

for t in times
    tv = tvd(bridge_counts[t], Nbridge, step_counts[t], Nstep)
    @show t, tv
end


=#


# ─────────────────────────────────────────────────────────────────────────────
# Guide and step (Euler) for UniformDiscretePoissonIndelProcess
# ─────────────────────────────────────────────────────────────────────────────

"""
    Guide(P::UniformDiscretePoissonIndelProcess, t, Xt::DiscreteState, X1::DiscreteState)

Convenience wrapper that computes full-tensor Doob rates and wraps them in a
`Guide` whose payload is a NamedTuple (sub, del, ins).
"""
function Guide(P::UniformDiscretePoissonIndelProcess{T}, t::Real,
               Xt::DiscreteState{<:AbstractArray{<:Signed}},
               X1::DiscreteState{<:AbstractArray{<:Signed}}) where {T}
    rates = doob_ud_full_tensors_fast(P, tensor(Xt), tensor(X1), T(t))
    return Flowfusion.Guide((sub = rates.sub, del = rates.del, ins = rates.ins))
end


"""
    Guide(P::UniformDiscretePoissonIndelProcess, tvec, Xts, X1s)

Build a batched Guide of full-tensor Doob rates for a batch of current states `Xts`
and targets `X1s` at times `tvec`.

Returns `Guide((sub, del, ins))` where shapes are:
- sub :: (K, nmax, B)
- del :: (1, nmax, B)
- ins :: (K, nmax+1, B)
Unfilled positions are zero and should be masked by the loss using letter/gap masks.
"""
function Guide(P::UniformDiscretePoissonIndelProcess{T},
               tvec::AbstractVector{<:Real},
               Xts::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
               X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}}) where {T}
    B = length(Xts)
    @assert B == length(X1s) == length(tvec)
    lens = length.(tensor.(Xts))
    nmax = maximum(lens)
    K = P.k
    # Match Guide dtype to tvec eltype to keep losses type-consistent with model
    S = eltype(tvec)
    sub = zeros(S, K, nmax, B)
    del = zeros(S, 1, nmax, B)
    ins = zeros(S, K, nmax + 1, B)
    lmask = falses(nmax, B)
    gapmask = falses(nmax + 1, B)
    for b in 1:B
        t = T(tvec[b])
        Xt = Xts[b]
        X1 = X1s[b]
        rates = doob_ud_full_tensors_fast(P, tensor(Xt), tensor(X1), t)
        n = length(tensor(Xt))
        sub[:, 1:n, b] .= S.(rates.sub)
        del[1, 1:n, b] .= S.(rates.del)
        ins[:, 1:(n+1), b] .= S.(rates.ins)
        lmask[1:n, b] .= true
        gapmask[1:n+1, b] .= true
    end
    return Flowfusion.Guide((sub = sub, del = del, ins = ins), gapmask, lmask)
end


function GuideAligned(P::UniformDiscretePoissonIndelProcess{T}, t::Real,
                      Xt::DiscreteState{<:AbstractArray{<:Signed}},
                      X1::DiscreteState{<:AbstractArray{<:Signed}};
                      rng::AbstractRNG = Random.default_rng()) where {T}

    C = make_precomp(P, T(t))
    B = make_branch_ud(P, C)
    Xt_vec = collect(tensor(Xt))
    x1_vec = collect(tensor(X1))

    events, _ = sample_alignment_branch_ud(rng, Xt_vec, x1_vec, B)
    rates = doob_ud_full_tensors_aligned(P, Xt_vec, x1_vec, T(t), events)
    return Flowfusion.Guide((sub = rates.sub, del = rates.del, ins = rates.ins))
end

function GuideAligned2(P::UniformDiscretePoissonIndelProcess{T}, t::Real,
  Xt::DiscreteState{<:AbstractArray{<:Signed}},
  X1::DiscreteState{<:AbstractArray{<:Signed}};
  rng::AbstractRNG = Random.default_rng()) where {T}

  C = make_precomp(P, T(t))
  B = make_branch_ud(P, C)
  Xt_vec = collect(tensor(Xt))
  x1_vec = collect(tensor(X1))

  events, _ = sample_alignment_branch_ud(rng, Xt_vec, x1_vec, B)
  rates = doob_ud_full_tensors_aligned(P, Xt_vec, x1_vec, T(t), events)
  return Flowfusion.Guide((sub = rates.sub, del = rates.del, ins = rates.ins)), events
end



function GuideAligned2(P::UniformDiscretePoissonIndelProcess{T},
  tvec::AbstractVector{<:Real},
  Xts::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
  X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}};
  rng::AbstractRNG = Random.default_rng()) where {T}

    Bsz = length(Xts)
    @assert Bsz == length(X1s) == length(tvec)
    lens  = length.(tensor.(Xts))
    nmax  = maximum(lens)
    K     = P.k
    S     = eltype(tvec)

    events_vec = Vector{Vector{Event}}()
    sub    = zeros(S, K, nmax, Bsz)
    del    = zeros(S, 1, nmax, Bsz)
    ins    = zeros(S, K, nmax + 1, Bsz)
    lmask  = falses(nmax, Bsz)
    gapmask= falses(nmax + 1, Bsz)


    for b in 1:Bsz
      t = T(tvec[b])
      Xt = Xts[b]; X1 = X1s[b]
      C  = make_precomp(P, t)
      B  = make_branch_ud(P, C)
      Xt_vec = collect(tensor(Xt))
      x1_vec = collect(tensor(X1))
      events, _ = sample_alignment_branch_ud(rng, Xt_vec, x1_vec, B)
      push!(events_vec, events)
      rates = doob_ud_full_tensors_aligned(P, Xt_vec, x1_vec, t, events)
      n = length(Xt_vec)
      sub[:, 1:n, b]     .= S.(rates.sub)
      del[1, 1:n, b]     .= S.(rates.del)
      ins[:, 1:(n+1), b] .= S.(rates.ins)
      lmask[1:n, b]      .= true
      gapmask[1:n+1, b]  .= true
    end

    return Flowfusion.Guide((sub = sub, del = del, ins = ins), gapmask, lmask), events_vec
end


function GuideAligned(P::UniformDiscretePoissonIndelProcess{T},
                      tvec::AbstractVector{<:Real},
                      Xts::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}},
                      X1s::Vector{<:DiscreteState{<:AbstractArray{<:Signed}}};
                      rng::AbstractRNG = Random.default_rng()) where {T}

    Bsz = length(Xts)
    @assert Bsz == length(X1s) == length(tvec)
    lens  = length.(tensor.(Xts))
    nmax  = maximum(lens)
    K     = P.k
    S     = eltype(tvec)

    sub    = zeros(S, K, nmax, Bsz)
    del    = zeros(S, 1, nmax, Bsz)
    ins    = zeros(S, K, nmax + 1, Bsz)
    lmask  = falses(nmax, Bsz)
    gapmask= falses(nmax + 1, Bsz)

    for b in 1:Bsz
        t = T(tvec[b])
        Xt = Xts[b]; X1 = X1s[b]
        C  = make_precomp(P, t)
        B  = make_branch_ud(P, C)
        Xt_vec = collect(tensor(Xt))
        x1_vec = collect(tensor(X1))
        events, _ = sample_alignment_branch_ud(rng, Xt_vec, x1_vec, B)
        rates = doob_ud_full_tensors_aligned(P, Xt_vec, x1_vec, t, events)
        n = length(Xt_vec)
        sub[:, 1:n, b]     .= S.(rates.sub)
        del[1, 1:n, b]     .= S.(rates.del)
        ins[:, 1:(n+1), b] .= S.(rates.ins)
        lmask[1:n, b]      .= true
        gapmask[1:n+1, b]  .= true
    end

    return Flowfusion.Guide((sub = sub, del = del, ins = ins), gapmask, lmask)
end

"""
    step(P::UniformDiscretePoissonIndelProcess, Xt, guide::Guide, s1, s2)

Takes a short stochastic step under instantaneous hazard rates provided in
`guide.H` as NamedTuple (sub, del, ins):
- sub :: (K, n) per-position substitution hazards (self-sub zero)
- del :: (n,) deletion hazards per position
- ins :: (K, n+1) per-gap insertion hazards

Only implemented for 1D `DiscreteState` without batch.
"""
function step(P::UniformDiscretePoissonIndelProcess,
              Xt::DiscreteState{<:AbstractArray{<:Signed}},
              hat,
              s1::Real, s2::Real)
    # Extract hazards
    @assert ndims(Xt.state) == 1 "UniformDiscretePoissonIndelProcess.step only supports 1D DiscreteState"
    # Apply process transform and zero self-substitutions before stepping
    sub = Array(P.transform(hat.sub)[:, :, 1])  # (K, n)
    del = vec(Array(P.transform(hat.del))[1, :, 1])  # (n,)
    ins = Array(P.transform(hat.ins)[:, :, 1])  # (K, n+1)
    K, n = size(hat.sub)
    # Zero self substitutions using current tokens
    if size(tensor(Xt), 1) > 0
      current_mask = tensor(onehot(Xt))[:,:,1] #onehotbatch(Xt_raw, 1:K)   # (K, n, 1)
      sub .= sub .* (1 .- current_mask)
    end
    @assert length(del) == n
    @assert size(ins, 1) == K && size(ins, 2) == n + 1
    dt = float(s2 - s1)
    x = collect(tensor(Xt))  # Vector{Int}
    # Site events: choose at most one per site using thinning
    to_delete = falses(n)
    sub_to = zeros(Int, n)    # 0 => no substitution; otherwise token id
    for i in 1:n
        r_del = del[i]
        r_sub_total = sum(@view sub[:, i])
        r_tot = r_del + r_sub_total
        if r_tot > 0
            p = 1 - exp(-dt * r_tot)
            if rand() < p
                u = rand() * r_tot
                if u < r_del
                    to_delete[i] = true
                else
                    u2 = u - r_del
                    # choose token by cumulative sum over sub[:, i]
                    acc = 0.0
                    chosen = 0
                    @inbounds for tok in 1:K
                        acc += sub[tok, i]
                        if u2 <= acc
                            chosen = tok
                            break
                        end
                    end
                    # In case of tiny numerical issues, fall back to last non-self token
                    chosen == 0 && (chosen = findfirst(!=(x[i]), 1:K) |> something(1))
                    sub_to[i] = chosen
                end
            end
        end
    end

    # Gap insertions: at most one per gap using thinning relative to original positions
    ins_tok = fill(0, n + 1)  # 0 => no insertion; otherwise token id
    for s in 0:n
        r_ins_total = sum(@view ins[:, s + 1])
        if r_ins_total > 0
            p = 1 - exp(-dt * r_ins_total)
            if rand() < p
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

    # Build new sequence by scanning original positions, applying deletions/subs and insertions at gaps
    result = Int[]
    # s = 0 gap
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
    sample_alignment_branch_ud(rng, Xt, x1, B) -> (events::Vector{Event}, h)

Sample a right-branch alignment path between Xt (root at t) and x1 (leaf at 1)
using BranchUD transitions. Returns events and total mass h.
"""
function sample_alignment_branch_ud(rng::AbstractRNG,
                                    Xt::Vector{Int},
                                    x1::Vector{Int},
                                    B::BranchUD{T}) where {T}
    n, m = length(Xt), length(x1)
    F = zeros(T, n+1, m+1); F[1,1] = one(T)

    # forward DP
    for i in 0:n, j in 0:m
        f = F[i+1, j+1]
        if i < n
            F[i+2, j+1] += f * (one(T) - B.surv)               # :A
        end
        if j < m
            F[i+1, j+2] += f * B.Iins                          # :B
        end
        if i < n && j < m
            P = (Xt[i+1] == x1[j+1]) ? B.Pdiag : B.Poff        # :R
            F[i+2, j+2] += f * B.surv * P
        end
    end
    h = F[n+1, m+1]

    # backward sample
    events = Vector{Event}()
    i, j = n, m
    while i>0 || j>0
        wA = (i>0)        ? F[i,   j+1] * (one(T) - B.surv) : zero(T)
        wB = (j>0)        ? F[i+1, j  ] * B.Iins            : zero(T)
        wR = (i>0 && j>0) ? F[i,   j  ] * B.surv * ((Xt[i]==x1[j]) ? B.Pdiag : B.Poff) : zero(T)
        wT = wA + wB + wR

        if !(wT > 0)
            if i>0 && j>0
                push!(events, Event(:R, i, j)); i -= 1; j -= 1
            elseif i>0
                push!(events, Event(:A, i, 0)); i -= 1
            else
                push!(events, Event(:B, 0, j)); j -= 1
            end
            continue
        end
        u = rand(rng) * float(wT)
        if u < wA
            push!(events, Event(:A, i, 0));  i -= 1
        elseif u < wA + wB
            push!(events, Event(:B, 0, j));  j -= 1
        else
            push!(events, Event(:R, i, j));  i -= 1; j -= 1
        end
    end
    reverse!(events)
    return events, h
end



# ─────────────────────────────────────────────────────────────────────────────
# DP-free pseudo-marginal MH for insertions (no DP tables)
# ─────────────────────────────────────────────────────────────────────────────

# Unbiased IS estimate of h(root -> leaf) on the right branch using locally
# normalized step weights. Works in log-domain for stability.
function _h_estimate_is(root::Vector{Int}, leaf::Vector{Int}, B::BranchUD{T};
                        N::Int=64, rng::AbstractRNG=Random.default_rng()) where {T}
    n, m = length(root), length(leaf)
    # log-sum-exp accumulator across N paths
    maxlog = -Inf
    acc = zero(T)

    @inbounds for _ in 1:N
        i = 0; j = 0
        logW = zero(T)
        while i < n || j < m
            wA = (i < n) ? (one(T) - B.surv) : zero(T)
            wB = (j < m) ?  B.Iins          : zero(T)
            wR = (i < n && j < m) ?
                 (B.surv * ((root[i+1] == leaf[j+1]) ? B.Pdiag : B.Poff)) : zero(T)

            Z = wA + wB + wR
            if !(Z > 0)  # underflow fallback: take any valid move, no weight update
                if i < n && j < m
                    i += 1; j += 1
                elseif i < n
                    i += 1
                else
                    j += 1
                end
                continue
            end

            u = rand(rng) * float(Z)
            if u < wA
                i += 1
            elseif u < wA + wB
                j += 1
            else
                i += 1; j += 1
            end
            logW += log(Z)
        end

        if logW > maxlog
            acc *= exp(maxlog - logW)
            maxlog = logW
        end
        acc += one(T)
    end

    return exp(maxlog) * (acc / T(N))
end

# Same estimator but with one symbol c inserted at gap s (0..n)
function _h_insert_estimate_is(root::Vector{Int}, s::Int, c::Int,
                               leaf::Vector{Int}, B::BranchUD{T};
                               N::Int=64, rng::AbstractRNG=Random.default_rng()) where {T}
    n = length(root)
    root2 = Vector{Int}(undef, n + 1)
    @inbounds begin
        for i in 1:s; root2[i] = root[i]; end
        root2[s+1] = c
        for i in s+1:n; root2[i+1] = root[i]; end
    end
    return _h_estimate_is(root2, leaf, B; N=N, rng=rng)
end

"""
    step_mh_pseudomarginal(P, Xt, hat, X1, s1, s2; N_is=64, rng)

Like your `step`, but **DP-free** and with **pseudo-marginal MH** on insertions:
- Propose insertions from `hat.ins` (aligned guide).
- Accept each proposed (gap s, token c) with
      α = min(1, r_true_hat / r_prop),
  where  r_true_hat = (λ/K) * ĥ_after / ĥ_before
  and ĥ_* are unbiased IS estimates from `_h_estimate_is` (no DP).

Substitutions and deletions use `hat.sub`/`hat.del` as in your `step`.
"""
function step_mh_pseudomarginal(P::UniformDiscretePoissonIndelProcess,
                                Xt::DiscreteState{<:AbstractArray{<:Signed}},
                                hat,
                                X1::DiscreteState{<:AbstractArray{<:Signed}},
                                s1::Real, s2::Real;
                                N_is::Int=64,
                                rng::AbstractRNG=Random.default_rng())

    @assert ndims(Xt.state) == 1 "UniformDiscretePoissonIndelProcess.step_mh_pseudomarginal only supports 1D DiscreteState"

    # Transform hazards and zero self-substitutions (same as your `step`)
    sub = Array(P.transform(hat.sub)[:, :, 1])             # (K, n)
    del = vec(Array(P.transform(hat.del))[1, :, 1])        # (n,)
    ins = Array(P.transform(hat.ins)[:, :, 1])             # (K, n+1)

    x      = collect(tensor(Xt))
    x1_vec = collect(tensor(X1))
    K, n   = size(sub, 1), size(sub, 2)

    if n > 0
        current_mask = tensor(onehot(Xt))[:,:,1]
        sub .= sub .* (1 .- current_mask)  # forbid self-subs
    end
    @assert length(del) == n
    @assert size(ins, 1) == K && size(ins, 2) == n + 1

    dt = float(s2 - s1)

    # ── Site events (delete / substitute): identical to your `step`
    to_delete = falses(n)
    sub_to    = zeros(Int, n)   # 0 => no substitution; otherwise token id
    for i in 1:n
        r_del = del[i]
        r_sub_total = sum(@view sub[:, i])
        r_tot = r_del + r_sub_total
        if r_tot > 0 && rand(rng) < (1 - exp(-dt * r_tot))
            u = rand(rng) * r_tot
            if u < r_del
                to_delete[i] = true
            else
                u2 = u - r_del
                acc = 0.0
                chosen = 0
                @inbounds for tok in 1:K
                    acc += sub[tok, i]
                    if u2 <= acc; chosen = tok; break; end
                end
                chosen == 0 && (chosen = findfirst(!=(x[i]), 1:K) |> something(1))
                sub_to[i] = chosen
            end
        end
    end

    # ── Gap insertions: propose under aligned guide; pseudo-marginal accept
    # Build right-branch constants once at left edge of the step (no DP tables)
    T = promote_type(Float64, eltype(sub), eltype(del), eltype(ins))
    C = make_precomp(P, T(s1))
    B = make_branch_ud(P, C)

    ins_tok = fill(0, n + 1)
    for s in 0:n
        r_ins_total = sum(@view ins[:, s + 1])
        if r_ins_total > 0 && rand(rng) < (1 - exp(-dt * r_ins_total))
            # propose token c ∝ ins[:, s+1]
            u = rand(rng) * r_ins_total
            acc = 0.0
            chosen = 0
            @inbounds for tok in 1:K
                acc += ins[tok, s + 1]
                if u <= acc; chosen = tok; break; end
            end
            chosen == 0 && (chosen = 1)

            # pseudo-marginal acceptance with coupled randomness to reduce variance
            seed = rand(rng, UInt)
            rng0 = Random.MersenneTwister(seed)
            rng1 = Random.MersenneTwister(seed)

            h0_hat = _h_estimate_is(x, x1_vec, B; N=N_is, rng=rng0)
            h1_hat = _h_insert_estimate_is(x, s, chosen, x1_vec, B; N=N_is, rng=rng1)

            r_true_hat = (P.λ / P.k) * (h1_hat / h0_hat)
            r_prop     = ins[chosen, s + 1]
            α = r_prop > 0 ? min(1.0, float(r_true_hat / r_prop)) : 0.0

            if rand(rng) < α
                ins_tok[s + 1] = chosen
            end
        end
    end

    # ── Build new sequence (same as your `step`)
    result = Int[]
    if ins_tok[1] != 0; push!(result, ins_tok[1]); end
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