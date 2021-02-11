### Our own implementation of the rank-revealing QR factorization.
# TODO: long-term replace this routine with something optimized, based on BLAS operations
#
# Written by Boris Bonev, Nov. 2020

using LinearAlgebra

# generate convenience access functions that copies A
prrqr(A::Matrix{T}, tol; reltol=false) where T = prrqr!(copy(A), tol; reltol=reltol)

# Utility routine to provide access to pivoted rank-revealing qr
function _compress_block!(A::Matrix{T}; tol::Real, reltol::Bool) where T
  Q, R, p = prrqr!(copy(A), tol; reltol)
  rk = min(size(R)...)
  return Q[:,1:rk], R[1:rk, invperm(p)]
end

# method for computing the pivoted rank-revealing qr in place
function prrqr!(A::Matrix{T}, tol::Real; reltol=false::Bool) where T
  m, n = size(A)

  vnrm = sum(abs2, A, dims=1)
  nrm = sum(vnrm)

  perm = collect(1:n)
  jj = 0

  # storage for Householder reflectors
  Pu = Vector{Vector{T}}(undef, min(m,n));
  Pb = Vector{T}(undef, min(m,n))

  for j = 1:min(m,n-1)
    # find the maximum and move it to the front
    mx, l = findmax(vnrm[j:end])
    vnrm[l+j-1] = vnrm[j]
    vnrm[j] = mx

    # figure out whether to stop the rrqr
    if ( reltol && sum(vnrm[j:end]) < sum(vnrm[1:j-1])*tol^2 ) || ( sum(vnrm[j:end]) < tol^2 )
      A = A[1:jj,:]
      break
    end

    # remember the permutation
    t = perm[l+j-1]
    perm[j+l-1] = perm[j]
    perm[j] = t

    # update A by swapping columns
    w = A[:,l+j-1]
    A[:,l+j-1] = A[:,j]
    A[:,j] = w
    w = w[j:end]

    # Compute the Householder reflector for w and apply it to A
    b, u = householder_reflector(w)
    A[j:end, j:end] = A[j:end,j:end] - b * u * (u' * A[j:end,j:end])

    # store the reflectors
    Pu[j] = u
    Pb[j] = b

    # Update the cvector norms
    for k = j+1:n
      vnrm[k] = sum(abs2, A[j+1:end,k]);
    end
    jj = j
  end

  Q = Matrix{T}(I, m, m)
  for j = jj:-1:1
    Q[j:end,:] = Q[j:end,:] - Pb[j] * Pu[j] * (Pu[j]' * Q[j:end,:]);
  end

  return Q, A, perm
end

function householder_reflector(w)
  u = w
  b = norm(u)
  s = sign(u[1])
  s == 0 ? s=1 : s=s
  u[1] = u[1] + b * s
  b = 2 / dot(u, u)
  return b, u
end