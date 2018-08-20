

function softthresh(beta::AbstractVector{<:Real},
                    penalty::AbstractVector{<:Real})
    [max(bt - lam, 0.0) * sign(bt) for (bt, lam) in zip(beta, penalty)]
end
