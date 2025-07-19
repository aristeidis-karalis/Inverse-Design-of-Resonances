
##############################################################################
## 
## Allocations for AllocatedLevenbergMarquardt
##
##############################################################################

struct AllocatedLevenbergMarquardt{Tx1, Tx2, Ty1, Ty2} <: AbstractAllocatedOptimizer
    δx::Tx1
    dtd::Tx2
    ftrial::Ty1
    fpredict::Ty2
    function AllocatedLevenbergMarquardt{Tx1, Tx2, Ty1, Ty2}(δx, dtd, ftrial, fpredict) where {Tx1, Tx2, Ty1, Ty2}
        length(dtd) == length(δx) || length(dtd) == length(ftrial) || throw(DimensionMismatch("The length of dtd must match either δx or ftrial."))
        length(ftrial) == length(fpredict) || throw(DimensionMismatch("The lengths of ftrial and fpredict must match."))
        new(δx, dtd, ftrial, fpredict)
    end
end

function AllocatedLevenbergMarquardt(δx::Tx1, dtd::Tx2, ftrial::Ty1, fpredict::Ty2) where {Tx1, Tx2, Ty1, Ty2}
    AllocatedLevenbergMarquardt{Tx1, Tx2, Ty1, Ty2}(δx, dtd, ftrial, fpredict)
end


function AbstractAllocatedOptimizer(nls::LeastSquaresProblem, optimizer::LevenbergMarquardt)
    z = (optimizer.solver.useTranspose && length(nls.y) < length(nls.x)) ? nls.y : nls.x
    AllocatedLevenbergMarquardt(_zeros(nls.x), _zeros(z), _zeros(nls.y), _zeros(nls.y))
end

##############################################################################
## 
## Optimizer for AllocatedLevenbergMarquardt
##
##############################################################################
##############################################################################


const MAX_Δ = 1e16 # minimum trust region radius
const MIN_Δ = 1e-16 # maximum trust region radius
const MIN_STEP_QUALITY = 1e-3
const GOOD_STEP_QUALITY = 0.75
const MIN_DIAGONAL = 1e-6
const MAX_DIAGONAL = 1e32

function optimize!(
    anls::LeastSquaresProblemAllocated{Tx, Ty, Tf, TJ, Tg, Toptimizer, Tsolver};
            x_tol::Number = 1e-8, f_tol::Number = 1e-8, g_tol::Number = 1e-8,
            iterations::Integer = 1_000, Δ::Number = 10.0, store_trace = false, show_trace = false, show_every = 1, lower = Vector{eltype(Tx)}(undef, 0), upper = Vector{eltype(Tx)}(undef, 0)) where {Tx, Ty, Tf, TJ, Tg, Toptimizer <: AllocatedLevenbergMarquardt, Tsolver}

    δx, dtd = anls.optimizer.δx, anls.optimizer.dtd
    ftrial, fpredict = anls.optimizer.ftrial, anls.optimizer.fpredict
    x, fcur, f!, J, g! = anls.x, anls.y, anls.f!, anls.J, anls.g!

    #is empty
    ((isempty(lower) || length(lower)==length(x)) && (isempty(upper) || length(upper)==length(x))) ||
            throw(ArgumentError("Bounds must either be empty or of the same length as the number of parameters."))
    ((isempty(lower) || all(x .>= lower)) && (isempty(upper) || all(x .<= upper))) || throw(ArgumentError("Initial guess must be within bounds."))

    decrease_factor = 2.0
    # initialize
    f_calls,  g_calls, mul_calls = 0, 0, 0
    converged, x_converged, f_converged, g_converged, converged =
        false, false, false, false, false
    f!(fcur, x)
    f_calls += 1
    ssr = sum(abs2, fcur)
    maxabs_gr = Inf
    need_jacobian = true

    eTx, eTy = eltype(x), eltype(fcur)

    iter = 0
    rcJ = max(size(J)...)
    Ndtd = length(dtd)

    tr = OptimizationTrace()
    tracing = store_trace || show_trace
    tracing && update!(tr, iter, ssr, maxabs_gr, store_trace, show_trace, show_every)

    while !converged && iter < iterations 
        iter += 1
        check_isfinite(x)
        
        # compute step
        if need_jacobian
        g!(J, x)
            g_calls += 1
            need_jacobian = false
        end

        dtd_value = norm(J)^2 / rcJ / Δ
        for i = 1:Ndtd; dtd[i] = dtd_value; end
        copyto!(fpredict, fcur)
        δx, maxabs_gr, lmiter = ldiv!(δx, J, fpredict, dtd,  anls.solver)
        mul_calls += lmiter

        # apply box constraints
        if !isempty(lower)
            @simd for i in 1:length(x)
               @inbounds δx[i] = min(δx[i], x[i] - lower[i])
            end
        end
        if !isempty(upper)
            @simd for i in 1:length(x)
               @inbounds δx[i] = max(δx[i], x[i] - upper[i])
            end
        end
        
        # predicted ssr
        mul!(fpredict, J, δx)
        mul_calls += 1
        axpy!(-one(eTy), fcur, fpredict)
        predicted_ssr = sum(abs2, fpredict)

        #update x
        axpy!(-one(eTx), δx, x)
            
        # calculate f
        f!(ftrial, x)
        f_calls += 1

        # trial ssr
        trial_ssr = sum(abs2, ftrial)

        # check convergence
        x_converged, f_converged, g_converged, converged =
            assess_convergence(δx, x, maxabs_gr, ssr, trial_ssr, x_tol, f_tol, g_tol)

        # decision for next step
        ρ = (ssr - trial_ssr) / abs(ssr - predicted_ssr)
            
        if ρ > MIN_STEP_QUALITY
            copyto!(fcur, ftrial)
            ssr = trial_ssr
            # increase trust region radius (from Ceres solver)
            Δ = min(Δ / max(1/3, 1.0 - (2.0 * ρ - 1.0)^3), MAX_Δ)
            decrease_factor = 2.0
            need_jacobian = true
        else
            # revert update
            axpy!(one(eTx), δx, x)
            Δ = max(Δ / decrease_factor , MIN_Δ)
            decrease_factor *= 2.0
        end
        tracing && update!(tr, iter, ssr, maxabs_gr, store_trace, show_trace, show_every)
    end
    LeastSquaresResult("LevenbergMarquardt", x, ssr, iter, converged,
                        x_converged, x_tol, f_converged, f_tol, g_converged, g_tol, tr,
                        f_calls, g_calls, mul_calls)
end
