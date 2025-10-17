using Test
using Flowfusion


function transition_mask_from_Xt2(P::Flowfusion.EditFlow, Xt::AbstractMatrix{<:Integer})
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

function remaining_edits3(P::Flowfusion.EditFlow, Zt::AbstractMatrix{Int}, Z1::AbstractMatrix{Int}; dense::Bool=false)
    @assert size(Zt) == size(Z1)
    L, B = size(Z1)
    tokens        = P.k
    latent_token  = P.latent_token
    padding_token = P.padding_token

    # position indices per (i,j); shape (L,B)
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)
    batch_length = maximum(vec(pos[end, :]))

    # Inserts: only valid token rows 1..tokens
    ins_mask = (Zt .== latent_token) .& (1 .<= Z1 .<= tokens)
    ins_idx  = findall(ins_mask)
    ins_rows = Z1[ins_idx]
    ins_cols = pos[ins_idx]
    ins_b    = [I[2] for I in ins_idx]
    insert_edits = zeros(Float32, tokens, batch_length, B)
    @inbounds for i in eachindex(ins_rows)
        insert_edits[ins_rows[i], ins_cols[i], ins_b[i]] += 1f0
    end
    dense_inserts = (ins_rows, ins_cols, ins_b)

    # Subs: valid tokens 1..tokens, changed, non-latent on both sides
    sub_mask = (Zt .!= latent_token) .& (Z1 .!= latent_token) .& (Z1 .!= Zt) .& (1 .<= Z1 .<= tokens)
    sub_idx  = findall(sub_mask)
    sub_rows = Z1[sub_idx]
    sub_cols = pos[sub_idx]
    sub_b    = [I[2] for I in sub_idx]
    sub_edits = zeros(Float32, tokens, batch_length, B)
    @inbounds for i in eachindex(sub_rows)
        sub_edits[sub_rows[i], sub_cols[i], sub_b[i]] = 1f0
    end
    dense_subs = (sub_rows .+ tokens, sub_cols, sub_b)

    # Dels: latent in Z1
    del_mask = (Z1 .== latent_token)
    del_idx  = findall(del_mask)
    del_cols = pos[del_idx]
    del_b    = [I[2] for I in del_idx]
    del_edits = zeros(Float32, 1, batch_length, B)
    @inbounds for i in eachindex(del_cols)
        del_edits[1, del_cols[i], del_b[i]] = 1f0
    end
    dense_dels = ((2*tokens+1).*ones(Int, length(del_cols)), del_cols, del_b)

    return dense ? (dense_inserts, dense_subs, dense_dels) : vcat(insert_edits, sub_edits, del_edits)
end
#=
function remaining_edits2(P::Flowfusion.EditFlow, Zt::Matrix{Int}, Z1::Matrix{Int}, dense=false)
    padding_token = P.padding_token
    latent_token = P.latent_token
    tokens = P.k
    (_, batch_size) = size(Z1)

    filtered_cols = [filter(x -> x != latent_token, col) for col in eachcol(Zt)]
    batch_length = maximum(length, filtered_cols) 
    pos = cumsum((Zt .!= padding_token) .& (Zt .!= latent_token), dims=1)

    #println("pos: ", pos)
    #print((Zt .!= padding_token) .& (Zt .!= latent_token))
    
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
@testset "EditFlow remaining_edits, transition mask, remove/pad, loss" begin
    # Match reference test parameters
    tokens = 21
    pad = 22
    lat = 23
    P = Flowfusion.EditFlow(tokens; transform=identity, padding_token=pad, latent_token=lat)

    # ─────────────────────────────────────────────────────────────────────
    # remaining_edits: simple 1-column case
    Zt = [0; 7; 23; 23; 4; 23; 22;;]
    Z1 = [0; 7; 20; 20; 4; 10; 22;;]
    expected = zeros(Float32, 2*tokens+1, 4, 1)
    expected[20,2,1] = 2
    expected[10,3,1] = 1
    got = remaining_edits3(P, Zt, Z1)

    display(findall(!iszero, got))
    display(got[20,2,1])
    display(got[10,3,1])
    display(size(got))
    display(size(expected))
    @test got == expected

    # Two-column case with inserts/subs
    Zt = [0 0; 7 23; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, 6, 2)
    expected[5, 4, 1] = 1
    expected[tokens+10,5,1] = 1
    expected[tokens+10,5,2] = 1
    expected[tokens+20, 4, 1] = 1
    expected[tokens+20, 3, 2] = 1
    expected[tokens+20, 3, 1] = 1
    expected[tokens+20, 2, 2] = 1
    expected[7, 1, 2] = 1
    got = remaining_edits2(P, Zt, Z1)
    @test got == expected

    # deletions case
    Zt = [0 0; 7 7; 20 20; 20 20; 4 4; 19 19; 22 22;]
    Z1 = [0 0; 7 7; 23 23; 20 23; 4 4; 23 23; 22 22;]
    expected = zeros(Float32, 2*tokens+1, 7, 2)
    expected[2*tokens+1,3,1] = 1
    expected[2*tokens+1,6,1] = 1
    expected[2*tokens+1,3,2] = 1
    expected[2*tokens+1,4,2] = 1
    expected[2*tokens+1,6,2] = 1
    got = remaining_edits2(P, Zt, Z1)
    @test got == expected

    # mixed inserts/subs case
    Zt = [0 0; 7 7; 15 15; 15 15; 23 4; 2 2; 22 22;]
    Z1 = [0 0; 7 7; 20 20; 20 20; 5 4; 10 10; 22 22;]
    expected = zeros(Float32, 2*tokens+1, 7, 2)
    expected[5, 4, 1] = 1
    expected[tokens+20,3,1] = 1
    expected[tokens+20,4,1] = 1
    expected[tokens+10,5,1] = 1
    expected[tokens+20,3,2] = 1
    expected[tokens+20,4,2] = 1
    expected[tokens+10,6,2] = 1
    got = remaining_edits2(P, Zt, Z1)
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # transition mask from Xt
    Xt = [0 0; 7 7; 20 20; 20 20; 5 4; 10 22; 22 22;]
    expected = ones(Float32, 2*tokens+1, size(Xt)...)
    # sample 1 (no self-sub mask for BOS=0)
    expected[tokens+7,2,1] = 0
    expected[tokens+20,3,1] = 0
    expected[tokens+20,4,1] = 0
    expected[tokens+5,5,1] = 0
    expected[tokens+10,6,1] = 0
    expected[:,7,1] .= 0
    expected[tokens+1:2*tokens+1,1,1] .= 0
    # sample 2 (no self-sub mask for BOS=0)
    expected[tokens+7,2,2] = 0
    expected[tokens+20,3,2] = 0
    expected[tokens+20,4,2] = 0
    expected[tokens+4,5,2] = 0
    expected[:,6:7,2] .= 0
    expected[tokens+1:2*tokens+1,1,2] .= 0


    got = transition_mask_from_Xt2(P, Xt)

    # Display all indices in `got` where the value is zero
    zero_indices = findall(x -> x == 0, got)
    #@info "Indices in 'got' that are zero:" zero_indices
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # remove_and_pad_concise
    Zt = [0 3 2;
          9 23 2;
          23 23 3;
          0 2 23]
    expected = [0 3 2;
                9 2 2;
                0 22 3]
    got = Flowfusion.remove_and_pad_concise(Zt, P.latent_token, P.padding_token)
    @test got == expected

    # ─────────────────────────────────────────────────────────────────────
    # loss equivalence under identity transform
    edit_multiplier = [0; 2;; 1; 0;;; 1; 0;; 0; 1;;;]     # (2,2,2)
    transition_mask = [1; 0;; 1; 0;;; 1; 0;; 0; 1;;;]     # (2,2,2)
    M = [0.1; 0.2;; 0.3; 0.4;;; 0.5; 0.6;; 0.7; 0.8;;;]   # (2,2,2)
    t = [0.3; 0.7;;]                                       # (1,2)
    k(t)=t; dk(t)=1
    scheduler_scaling = dk.(t) ./ (-k.(t) .+ 1)
    # manual loss
    l = (0.1+0.3+0.5+0.8 - (1/(1-0.3)*(2*log(0.2)+log(0.3)) + 1/(1-0.7)*(log(0.5)+log(0.8))))/2
    got = Flowfusion.edit_loss(P, M, transition_mask, edit_multiplier, scheduler_scaling; op_mask=nothing, eps=0)
    @test isapprox(got, l; atol=1e-7, rtol=1e-7)

end


