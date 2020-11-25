### This File contains all compression routines for the HssMatrices.jl package
#
# Direct compression
# as described in
# Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices.
# Numerical Linear Algebra with Applications, 17(6), 953–976. https://doi.org/10.1002/nla.691
#
# Recompression
# as described in
# Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices.
# Numerical Linear Algebra with Applications, 17(6), 953–976. https://doi.org/10.1002/nla.691
#
# Randomized compression
# as described in
# Martinsson, P. G. (2011). A Fast Randomized Algorithm for Computing a Hierarchically Semiseparable Representation of a Matrix.
# SIAM Journal on Matrix Analysis and Applications, 32(4), 1251–1274. https://doi.org/10.1137/100786617
#
# Written by Boris Bonev, Nov. 2020

## Direct compression algorithm
# wrapper function that will be exported
function hss_compress_direct(A::Matrix{T}, rcl::BinaryNode, ccl::BinaryNode, tol=tol; reltol=reltol) where T
  m = length(rcl.data); n = length(ccl.data)
  hssA = HssMatrix{T}(); hssA.rootnode = true;
  hssA = hss_from_cluster!(hssA, rcl, ccl)
  Brow = Array{T}(undef, m, 0)
  Bcol = Array{T}(undef, 0, n)
  hss_compress_direct!(hssA, A, Brow, Bcol, 0, 0, m, n, tol; reltol)
  return hssA
end

# recursive function
# TODO: think about typestability, promotion, etc. MAYBE this needs to be modified
function hss_compress_direct!(hssA::HssMatrix, A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, ro, co, m, n, tol; reltol) where T
  if hssA.leafnode
    hssA.D = A[ro+1:ro+m, co+1:co+n]
    # compress HSS block row/col
    hssA.U, Brow = compress_block!(Brow, tol; reltol)
    hssA.V, Bcol = compress_block!(copy(Bcol'), tol; reltol); Bcol = copy(Bcol') # quick fix to avoid Julia's lazy transpose
  else
    m1 = hssA.m1; n1 = hssA.n1; m2 = hssA.m2; n2 = hssA.n2

    # form blocks to be compressed in the children step
    Brow1 = hcat(A[ro+1:ro+m1, co+n1+1:co+n1+n2], Brow[1:m1, :]) #FIXME: rootnode?
    Bcol1 = vcat(A[ro+m1+1:ro+m1+m2, co+1:co+n1], Bcol[:, 1:n1])
    Brow1, Bcol1 = hss_compress_direct!(hssA.A11, A, Brow1, Bcol1, ro, co, m1, n1, tol; reltol)
    
    # form blocks to be compressed in the children stepnull
    Brow2 = hcat(Bcol1[1:m2, :], Brow[m1+1:end, :])
    Bcol2 = vcat(Brow1[:, 1:n2], Bcol[:, n1+1:end])
    Brow2, Bcol2 = hss_compress_direct!(hssA.A22, A, Brow2, Bcol2, ro+m1, co+n1, m2, n2, tol; reltol)

    # figure out dimensions of reduced blocks
    rm1 = size(Brow1,1); rn1 = size(Bcol1, 2)
    rm2 = size(Brow2,1); rn2 = size(Bcol2, 2)

    hssA.B12 = Bcol2[1:rm1, :]
    hssA.B21 = Brow2[:, 1:rn1]

    if !hssA.rootnode
      # clean up stuff from the front and form the composed HSS block row/col for compression
      Brow = vcat(Brow1[:, n2+1:end],  Brow2[:, rn1+1:end])
      Bcol = hcat(Bcol1[m2+1:end, :],  Bcol2[rm1+1:end, :])

      # do the actual compression and disentangle blocks of the translation operators
      R, Brow = compress_block!(Brow, tol; reltol)
      hssA.R1 = R[1:rm1, :]
      hssA.R2 = R[rm1+1:end, :]
    
      W, Bcol = compress_block!(copy(Bcol'), tol; reltol); Bcol = copy(Bcol')
      hssA.W1 = W[1:rn1, :]
      hssA.W2 = W[rn1+1:end, :]
    end
  end
  return Brow, Bcol
end

function compress_block!(A::Matrix{T}, tol; reltol) where T
  Q, R, p = prrqr!(A, tol; reltol)
  rk = size(R,1);
  return Q[:,1:rk], R[1:rk, invperm(p)]
end

## Recompression algorithm
function hss_recompress!(hssA::HssMatrix{T}, tol=tol; reltol=reltol) where T
  if hssA.leafnode; return hssA; end
  # a prerequisite for this algorithm to work is that generators are orthonormal
  orthonormalize_generators!(hssA)
  # define Brow, Bcol
  Brow = Array{T}(undef, 0, 0)
  Bcol = Array{T}(undef, 0, 0)
  hss_recompress_rec!(hssA, Brow, Bcol, tol; reltol)
end

# recursive definition
function hss_recompress_rec!(hssA::HssMatrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, tol=tol; reltol=reltol) where T
  if hssA.leafnode; error("recompression called on a leafnode"); end
  if hssA.rootnode
    # compress B12, B21 via something that resembles the SVD
    P1, S2 = compress_block!(hssA.B12, tol; reltol)
    Q2, T1 = compress_block!(copy(hssA.B12'), tol; reltol)
    P2, S1 = compress_block!(hssA.B21, tol; reltol)
    Q1, T2 = compress_block!(copy(hssA.B21'), tol; reltol)

    hssA.B12 = S2*Q2
    hssA.B21 = S1*Q1

    # pass information to children and proceed recursively
    if !hssA.A11.leafnode
      hssA.A11.R1 = hssA.A11.R1*P1
      hssA.A11.R2 = hssA.A11.R2*P1
      hssA.A11.W1 = hssA.A11.W1*Q1
      hssA.A11.W2 = hssA.A11.W2*Q1
    else
      hssA.A11.U = hssA.A11.U*P1
      hssA.A11.V = hssA.A11.V*Q1
    end
    # repeat for A22
    if !hssA.A22.leafnode
      hssA.A22.R1 = hssA.A22.R1*P2
      hssA.A22.R2 = hssA.A22.R2*P2
      hssA.A22.W1 = hssA.A22.W1*Q2
      hssA.A22.W2 = hssA.A22.W2*Q2
    else
      hssA.A22.U = hssA.A22.U*P2
      hssA.A22.V = hssA.A22.V*Q2
    end

    # call recompression recursively
    if !hssA.A11.leafnode
      hss_recompress_rec!(hssA.A11, hssA.B12, copy(hssA.B21'), tol; reltol)
    end
    if !hssA.A22.leafnode
      hss_recompress_rec!(hssA.A22, hssA.B21, copy(hssA.B12'), tol; reltol)
    end
  else
    # compress B12
    Brow1 = hcat(hssA.B12, hssA.R1*Brow)
    Bcol2 = hcat(hssA.B12', hssA.W2*Bcol)
    P1, S2 = compress_block!(Brow1, tol; reltol)
    Q2, T1 = compress_block!(Bcol2, tol; reltol)
    # get the original number of columns in B12
    rm1, rn2 = size(hssA.B12)
    hssA.B12 = S2[:,1:rn2]*Q2
    Brow1 = S2[:, rn2+1:end]
    Bcol2 = T1[:, rm1+1:end]
    # compress B21
    Brow2 = hcat(hssA.B21, hssA.R2*Brow)
    Bcol1 = hcat(hssA.B21', hssA.W1*Bcol)
    P2, S1 = compress_block!(Brow2, tol; reltol)
    Q1, T2 = compress_block!(Bcol1, tol; reltol)
    # get the original number of columns in B21
    rm2, rn1 = size(hssA.B21)
    hssA.B21 = S1[:,1:rn1]*Q1
    Brow2 = S1[:,rn1+1:end]
    Bcol1 = T2[:, rm2+1:end]

    # update generators of A11
    if !hssA.A11.leafnode
      hssA.A11.R1 = hssA.A11.R1*P1
      hssA.A11.R2 = hssA.A11.R2*P1
      hssA.A11.W1 = hssA.A11.W1*Q1
      hssA.A11.W2 = hssA.A11.W2*Q1
    else
      hssA.A11.U = hssA.A11.U*P1
      hssA.A11.V = hssA.A11.V*Q1
    end
    # update generators in A22
    if !hssA.A22.leafnode
      hssA.A22.R1 = hssA.A22.R1*P2
      hssA.A22.R2 = hssA.A22.R2*P2
      hssA.A22.W1 = hssA.A22.W1*Q2
      hssA.A22.W2 = hssA.A22.W2*Q2
    else
      hssA.A22.U = hssA.A22.U*P2
      hssA.A22.V = hssA.A22.V*Q2
    end

    # call recompression recursively
    if !hssA.A11.leafnode
      Brow1 = hcat(hssA.B12, S2[:,rn2+1:end])
      Bcol1 = hcat(hssA.B21', T2[:,rm2+1:end])
      hss_recompress_rec!(hssA.A11, Brow1, Bcol1, tol; reltol)
    end
    if !hssA.A22.leafnode
      Brow2 = hcat(hssA.B21, S1[:,rn1+1:end])
      Bcol2 = hcat(hssA.B12', T1[:,rm1+1:end])
      hss_recompress_rec!(hssA.A22, Brow2, Bcol2, tol; reltol)
    end
  end
end

# # imitates the interface to SVD, but uses prrqr
# function truncate_block!(A::Matrix{T}, tol; reltol) where T
#   Q, R, p = prrqr!(A, tol; reltol)
#   rk = size(R,1);
#   U = Q[:,1:rk]
#   S = R[1:rk, invperm(p)]
#   F = qr(S')
#   V = Matrix(F.Q)
#   S = copy(F.R')
#   return U, S, V
# end

## Randomized compression will go here...