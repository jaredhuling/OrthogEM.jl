
function oem_fit(x::AbstractMatrix{<:Real},
                 y::AbstractVector{<:Real};
                 intercept::Bool = true,
                 standardize::Bool = true,
                 nlambda::Int = 100,
                 maxit::Int = 1000,
                 tol::BLAS.BlasFloat = 1e-5,
                 lambda_min_ratio::BLAS.BlasFloat = 1e-3,
                 penalty_factor::AbstractVector{<:Real} = ones(Float64, size(x)[2]))

    n, p = size(x)

    if intercept
        p += 1
        penalty_factor = vcat(0.0, penalty_factor)
    end

    xtx, xty, colsquare_inv, intscale = compute_xtx_y(x, y, intercept, standardize)

    # get eigenvalues, stepsize, A
    steps, A, d = make_steps_A(xtx)

    # initialize beta as vector of zeros
    beta  = zeros(p)

    # compute maximum lambda value
    lambda_max = compute_maxlam(xty, penalty_factor)

    # compute sequence of values for lambda
    lambda = exp.(range(log(lambda_max),
                        stop   = log(lambda_min_ratio * lambda_max),
                        length = nlambda))

    beta_mat  = zeros(p, nlambda)
    iters_vec = repeat([maxit], outer = nlambda)

    # loop over each lambda value
    for l in 1:nlambda
        # set up penalty amounts
        penalty_val   = repeat([lambda[l]], outer = p)
        # adjust penalty by user-specified factors
        penalty_val .*= penalty_factor

        iters = maxit

        # start oem iterations
        for i in 1:maxit
            beta_prev = beta

            u = xty + A * beta_prev

            # M-step
            beta = softthresh(u, penalty_val) ./ steps

            #for j in 1:p
            #    if beta[j] != beta_prev[j]
            #        resid .+= x[:,j] * (beta[j] - beta_prev[j])
            #    end
            #end

            if all(abs.(beta - beta_prev) .<= tol)
                iters = i
                break
            end
        end # end loop over iterations

        ## un-standardize coefficients
        if standardize
            if intercept
                beta[2:p] .*= colsquare_inv[1,:]
            else
                beta .*= colsquare_inv[1,:]
            end
        end

        beta_mat[:,l] = beta
        iters_vec[l] = iters
    end # end loop over lambda


    if intercept
        beta_mat[1,:] ./= intscale
    end

    beta_mat, iters_vec, lambda, d
end
