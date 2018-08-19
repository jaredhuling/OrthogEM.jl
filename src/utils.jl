

compute_maxlam = function(xty::StridedVector{T},
                          penalty_factor::StridedVector{T})  where T <: BLAS.BlasFloat
    pp = length(penalty_factor)
    p = length(xty)

    #if pp + 1 == p
    #    deleteat!(penalty_factor, 1)
    #elseif pp != p
    #    error("wrong dimensions")
    #end

    if pp != p
        error("wrong dimensions")
    end

    # compute largest value for lambda
    mask = ones(p)
    mask[penalty_factor .== 0.0] .= 0.0
    pf2 = penalty_factor
    pf2[pf2 .== 0] .= 1.0
    mask ./= pf2
    lambda_max = max(xty .* mask...)
end

make_steps_A = function(xtx::AbstractMatrix{T}) where T <: BLAS.BlasFloat
    pp, p = size(xtx)
    d, v   = eigs(xtx, nev = 1)
    d      = reinterpret(Float64, d)[1]

    steps = repeat([d], outer = p)
    A     = Diagonal(steps) - xtx
    steps, A, d
end


compute_xtx_y = function(x::StridedMatrix{T},
                         y::StridedVector{T},
                         intercept::Bool = true,
                         standardize::Bool = true) where T <: BLAS.BlasFloat

    n, p = size(x)

    intscale = 1.0

    if standardize
        colsquare = sqrt.(sum(x .^ 2, dims = 1) / (n - 1))
        colsums   = sum(x, dims = 1) / n
    else
        colsquare = ones(p)
        colsums   = ones(p)
    end

    colsquare_inv = 1 ./ colsquare

    if intercept & standardize
        xtx = zeros(p + 1, p + 1)
        xty = zeros(p + 1)
        intscale = 1 / n

        colsums .*= colsquare_inv

        xtx[2:(p+1),2:(p+1)] = Diagonal(colsquare_inv[1,:]) * Symmetric(x'x) * Diagonal(colsquare_inv[1,:]) / n
        xtx[1, 2:(p+1)] = colsums * intscale
        xtx[2:(p+1), 1] = xtx[1, 2:(p+1)]'
        xtx[1, 1] = 1.0 * intscale
        xty[2:(p+1)] = ((x' * y) .* colsquare_inv') / n'
        xty[1] = mean(y) * intscale

    elseif intercept
        xtx = zeros(p + 1, p + 1)
        xty = zeros(p + 1)

        xtx[2:(p+1), 2:(p+1)] = Symmetric(x'x) / n
        xtx[1, 2:(p+1)] = colsums
        xtx[2:(p+1), 1] = xtx[1, 2:(p+1)]'
        xtx[1, 1] = 1.0

        xty[2:(p+1)] = x' * y / n'
        xty[1] = mean(y)
    elseif standardize
        xtx = Diagonal(colsquare_inv[1,:]) * Symmetric(x'x) * Diagonal(colsquare_inv[1,:]) / n
        xty = (((x' * y) .* colsquare_inv') / n')[:,1]
    else
        xtx = Symmetric(x'x) / n
        xty = x' * y / n'
    end
    Symmetric(xtx), xty, colsquare_inv, intscale
end



function create_steps(x::StridedMatrix{T},
                      group_col::StridedVector{Int}) where T <: BLAS.BlasFloat
    grps   = unique(group_col)
    n_grps = length(grps)
    n, p   = size(x)

    steps  = ones(p)
    p_cur = 0

    xtx_list = Any[]
    grp_idx_list = Any[]
    grp_bool_list = Any[]
    steps_list = Any[]
    eigens = zeros(n_grps)
    for j in 1:n_grps
        cur_grp = group_col .== grps[j]
        grp_len = sum(cur_grp)
        xc      = x[:,cur_grp]
        xtx_cur = Symmetric(xc'xc) / n

        d, v   = eigs(xtx_cur, nev = 1)

        push!(steps_list, repeat(d, outer = grp_len))
        push!(xtx_list, xtx_cur)
        push!(grp_bool_list, cur_grp)
        push!(grp_idx_list, findall(cur_grp))
        eigens[j] = d[1]
    end
    eigens, steps_list, xtx_list, grp_idx_list, grp_bool_list, grps, n_grps
end
