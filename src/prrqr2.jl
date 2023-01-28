### Our own implementation of the rank-revealing QR factorization.
# TODO: long-term replace this routine with something optimized, based on BLAS operations
#
# Householder reflectors as seen in
# Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations. Johns Hopkins University Press.
#
# Rank-revealing QR factorization as decribed in
# Gu, Eisenstat. EFFICIENT ALGORITHMS FOR COMPUTING A STRONG RANK-REVEALING QR FACTORIZATION. Siam J. Sci. Comput.
# 
# Re-Written by Boris Bonev, Feb. 2021

import LinearAlgebra.LAPACK.orgqr!

# generate convenience access functions that copies A
prrqr(A::Matrix{T}, atol, rtol) where T = _prrqr!(copy(A), atol, rtol)
prrqr!(A::Matrix{T}, atol, rtol) where T = _prrqr!(A, atol, rtol)
prrqr(A::Matrix{T}, atol, rtol) where T = prrqr!(copy(A), atol, rtol)
prrqr(A::Adjoint, atol, rtol) = prrqr(collect(A), atol, rtol)

# convenience access function# Utility routine to provide access to pivoted rank-revealing qr
function _compress_block!(A::Matrix{T}, atol::Float64, rtol::Float64) where T
  A, β, p = prrqr!(A, atol, rtol)
  k = length(β)
  Q = orgqr!(A[:,1:k], β, k)
  R = triu!(A[1:k, :])
  return Q, R[:, invperm(p)]
end

# method for computing the pivoted rank-revealing qr in place
# returns something similar to geqp3, in compact format
function _prrqr!(A::Matrix{T}, atol::Float64, rtol::Float64) where T
  m, n = size(A)
  isreal = T <: Real
  k = min(m, n)
  β = Array{T}(undef, k)
  jpvt = collect(1:n)

  #lwork = 2*n*isreal + (n + 1)*nb
  lwork = 2*n
  work = Array{T}(undef, lwork)

  #initialize column norms
  fnorm2 = 0.
  @inbounds for l = 1:n
    work[l] = work[n+l] = norm(A[:, l])
    fnorm2 += work[l]^2
  end

  # early exit if the norm is 0
  if fnorm2 == 0.
    return A, Array{T}(undef,0), jpvt
  end

  # compute tolerance to determine termination criterion
  ptol2 = max(rtol^2*fnorm2, atol^2)

  j = 1
  while j <= k
    maxnrm, l = findmax(@view work[j:n])
    # update the permutation vector and store the column to be exchanged
    jpvt[j], jpvt[l+j-1] = jpvt[l+j-1], jpvt[j]
    work[j], work[l+j-1] = work[l+j-1], work[j]
    # swap columns in place
    @inbounds for i = 1:m
      A[i,j], A[i,l+j-1] = A[i,l+j-1], A[i,j]
    end
    # compute the householder reflector in place
    v, β[j] = house!(A[j:end,j])
    A[j:end, j:end] .= A[j:end,j:end] .- β[j] .* v * (v' * A[j:end,j:end])
    #@inbounds for l = j:n
    # A[j:m, l] .=  A[j:m, l] .- β[j] * v * (v' * A[j:m, l])
    #end
    A[j+1:m,j] .= v[2:end]
    # update column norms
    fnorm2 = 0.
    @inbounds for l = j+1:n
      work[l] = norm(A[j+1:n,l]);
      fnorm2 += work[l]^2
    end
    if fnorm2 < ptol2
      break
    else
      j += 1
    end
  end
  rk = min(j,k);
  return A, β[1:rk], jpvt
end

# computes the householder reflector in place
function house!(x::Vector{T}) where T
  m = length(x); σ = dot(x[2:m],x[2:m]);
  if σ == 0. && x[1] >= 0.
    β = T(0.); x[1] = T(1.)
  elseif σ == 0. && x[1] < 0.
    β = T(-2.); x[1] = T(1.)
  else
    μ = sqrt(x[1]^2 + σ)
    if x[1] < 0.
      x[1] = x[1] - μ
    else
      x[1] = -σ/(x[1] + μ)
    end
    β = 2*x[1]^2/(σ + x[1]^2)
    x .= x./x[1]
  end
  return x, β
end
