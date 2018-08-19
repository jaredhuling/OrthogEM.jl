

function softthresh(beta::StridedVector{T},
                    penalty::StridedVector{T}) where T <: BLAS.BlasFloat
    p = length(beta)
    sign.(beta).*max.(beta - penalty, zeros(p))
end
