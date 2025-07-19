##############################################################################
## 
## Type with stored cholesky
##
##############################################################################

struct DenseCholeskyAllocatedSolver{Tc <: StridedMatrix} <: AbstractAllocatedSolver
    cholm::Tc
    function DenseCholeskyAllocatedSolver{Tc}(cholm) where {Tc <: StridedMatrix}
        size(cholm, 1) == size(cholm, 2) || throw(DimensionMismatch("chol must be square"))
        new(cholm)
    end
end
function DenseCholeskyAllocatedSolver(cholm::Tc) where {Tc <: StridedMatrix}
    DenseCholeskyAllocatedSolver{Tc}(cholm)
end


function AbstractAllocatedSolver(nls::LeastSquaresProblem{Tx, Ty, Tf, TJ, Tg}, optimizer::AbstractOptimizer{Cholesky{Bool}}) where {Tx, Ty, Tf, TJ <: StridedVecOrMat, Tg}
    cholm_size = optimizer.solver.useTranspose ? minimum(size(nls.J)) : length(nls.x)
    return DenseCholeskyAllocatedSolver(Array{eltype(nls.J)}(undef, cholm_size, cholm_size))
end

##############################################################################
## 
## solve J'J \ J'y or J'*[JJ' \ y] by Cholesky
##
##############################################################################

function LinearAlgebra.ldiv!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, A::DenseCholeskyAllocatedSolver)
    cholm = A.cholm
    NJ = size(cholm,1)
    
    if NJ == length(x)
        mul!(cholm, J',  J)
        mul!(x, J',  y)
        maxabs_gr = maximum(abs, x)   # maximum derivative (needed by the optimizer)
        ldiv!(cholesky!(Symmetric(cholm), Val(true)), x)
        
    elseif NJ == length(y)
        mul!(cholm, J,  J')
        ldiv!(cholesky!(Symmetric(cholm), Val(true)), y)
        mul!(x, J', y)
        maxabs_gr = maximum(abs, y)   # maximum derivative (needed by the optimizer)
    end
    return x, maxabs_gr, 1
    
end

##############################################################################
## 
## solve (J'J + diagm(damp)) \ J'y or J'*[(JJ' + diagm(damp)) \ y] by Cholesky
##
##############################################################################

function LinearAlgebra.ldiv!(x::AbstractVector, J::StridedMatrix, y::AbstractVector, 
            damp::AbstractVector, A::DenseCholeskyAllocatedSolver)
    cholm = A.cholm
    NJ = size(cholm, 1)
    NJ == length(damp) || throw(DimensionMismatch("size(chol, 1) should equal length(damp)"))
    
    if NJ == length(x)
        # update cholm as J'J + λdtd
        mul!(cholm, J', J)
        for i in 1:NJ; cholm[i, i] += damp[i]; end

        # solve
        mul!(x, J', y)
        maxabs_gr = maximum(abs, x)   # maximum derivative (needed by the optimizer)
        ldiv!(cholesky!(Symmetric(cholm)), x)
        
    elseif NJ == length(y)
        # update cholm as JJ' + λdtd
        mul!(cholm, J, J')
        for i in 1:NJ; cholm[i, i] += damp[i]; end

        # solve
        ldiv!(cholesky!(Symmetric(cholm)), y)
        mul!(x, J', y)
        maxabs_gr = maximum(abs, y)   # maximum derivative (needed by the optimizer)
    end        
    return x, maxabs_gr, 1

end

