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
function hss_compress_direct(A::Matrix{T}, rcl::ClusterTree, ccl::ClusterTree; tol=tol, reltol=reltol) where T
  m = length(rcl.data); n = length(ccl.data)
  if size(A) != (m,n) throw(ArgumentError("size of row- and column-cluster-trees must match")) end
  Brow = Array{T}(undef, m, 0)
  Bcol = Array{T}(undef, 0, n)
  if isleaf(rcl) && isleaf(ccl)
    hssA, _, _ = _compress_direct!(A, Brow, Bcol, rcl.data, ccl.data; tol, reltol)
  elseif isbranch(rcl) && isbranch(ccl)
    hssA, _, _ = _compress_direct!(A, Brow, Bcol, rcl, ccl; tol, reltol)
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end
  return hssA
end

# leaf node function for compression
function _compress_direct!(A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int}; tol, reltol) where T
  U, Brow = _compress_block!(Brow; tol, reltol)
  V, Bcol = _compress_block!(copy(Bcol'); tol, reltol) #TODO: write code that is better at dealing with Julia's lazy transpose
  return HssLeaf(A[rows, cols], U, V), Brow, copy(Bcol')
end

function _compress_direct!(A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, rcl::ClusterTree, ccl::ClusterTree; tol, reltol) where T
  m1 = length(rcl.left.data); m2 = length(rcl.right.data)
  n1 = length(ccl.left.data); n2 = length(ccl.right.data)

  rows1 = rcl.left.data; rows2 = rcl.right.data
  cols1 = ccl.left.data; cols2 = ccl.right.data

  # left node
  Brow1 = [A[rows1, cols2]  Brow[1:m1, :]]
  Bcol1 = [A[rows2, cols1]; Bcol[:, 1:n1]]
  if isleaf(rcl.left) && isleaf(ccl.left)
    A11, Brow1, Bcol1 = _compress_direct!(A, Brow1, Bcol1, rows1, cols1; tol, reltol)
  elseif isbranch(rcl.left) && isbranch(ccl.left)
    A11, Brow1, Bcol1 = _compress_direct!(A, Brow1, Bcol1, rcl.left, ccl.left; tol, reltol)
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end

  # right node
  Brow2 = [Bcol1[1:m2, :]  Brow[m1+1:end, :]]
  Bcol2 = [Brow1[:, 1:n2]; Bcol[:, n1+1:end]]
  if isleaf(rcl.right) && isleaf(ccl.right)
    A22, Brow2, Bcol2 = _compress_direct!(A, Brow2, Bcol2, rows2, cols2; tol, reltol)
  elseif isbranch(rcl.right) && isbranch(ccl.right)
    A22, Brow2, Bcol2 = _compress_direct!(A, Brow2, Bcol2, rcl.right, ccl.right; tol, reltol)
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end

  # figure out compressed off-diagonal blocks
  rm1 = size(Brow1,1); rn1 = size(Bcol1, 2)
  rm2 = size(Brow2,1); rn2 = size(Bcol2, 2)
  B12 = Bcol2[1:rm1, :]
  B21 = Brow2[:, 1:rn1]

  # clean up stuff from the front and form the composed HSS block row/col for compression
  Brow = [Brow1[:, n2+1:end]; Brow2[:, rn1+1:end]]
  Bcol = [Bcol1[m2+1:end, :]  Bcol2[rm1+1:end, :]]
  
  # do the actual compression and disentangle blocks of the translation operators
  R, Brow = _compress_block!(Brow; tol, reltol)
  R1 = R[1:rm1, :]
  R2 = R[rm1+1:end, :]
      
  W, Bcol = _compress_block!(copy(Bcol'); tol, reltol); Bcol = copy(Bcol')
  W1 = W[1:rn1, :]
  W2 = W[rn1+1:end, :]

  # call recursively the 
  hssA = HssNode(A11, A22, B12, B21, R1, W1, R2, W2)
  return hssA, Brow, Bcol
end

# # recursive function
# # TODO: think about typestability, promotion, etc. MAYBE this needs to be modified
# function hss_compress_direct!(hssA::HssMatrix, A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, ro::Integer, co::Integer, m::Integer, n::Integer, tol; reltol) where {T}
#   if hssA.leafnode
#     hssA.D = A[ro+1:ro+m, co+1:co+n]
#     # compress HSS block row/col
#     hssA.U, Brow = compress_block!(Brow, tol; reltol)
#     hssA.V, Bcol = compress_block!(copy(Bcol'), tol; reltol); Bcol = copy(Bcol') # quick fix to avoid Julia's lazy transpose
#   else
#     m1 = hssA.m1; n1 = hssA.n1; m2 = hssA.m2; n2 = hssA.n2

#     # form blocks to be compressed in the children step
#     Brow1 = hcat(A[ro+1:ro+m1, co+n1+1:co+n1+n2], Brow[1:m1, :]) #FIXME: rootnode?
#     Bcol1 = vcat(A[ro+m1+1:ro+m1+m2, co+1:co+n1], Bcol[:, 1:n1])
#     Brow1, Bcol1 = hss_compress_direct!(hssA.A11, A, Brow1, Bcol1, ro, co, m1, n1, tol; reltol)
    
#     # form blocks to be compressed in the children step
#     Brow2 = hcat(Bcol1[1:m2, :], Brow[m1+1:end, :])
#     Bcol2 = vcat(Brow1[:, 1:n2], Bcol[:, n1+1:end])
#     Brow2, Bcol2 = hss_compress_direct!(hssA.A22, A, Brow2, Bcol2, ro+m1, co+n1, m2, n2, tol; reltol)

#     # figure out dimensions of reduced blocks
#     rm1 = size(Brow1,1); rn1 = size(Bcol1, 2)
#     rm2 = size(Brow2,1); rn2 = size(Bcol2, 2)

#     hssA.B12 = Bcol2[1:rm1, :]
#     hssA.B21 = Brow2[:, 1:rn1]

#     if !hssA.rootnode
#       # clean up stuff from the front and form the composed HSS block row/col for compression
#       Brow = vcat(Brow1[:, n2+1:end],  Brow2[:, rn1+1:end])
#       Bcol = hcat(Bcol1[m2+1:end, :],  Bcol2[rm1+1:end, :])

#       # do the actual compression and disentangle blocks of the translation operators
#       R, Brow = compress_block!(Brow, tol; reltol)
#       hssA.R1 = R[1:rm1, :]
#       hssA.R2 = R[rm1+1:end, :]
    
#       W, Bcol = compress_block!(copy(Bcol'), tol; reltol); Bcol = copy(Bcol')
#       hssA.W1 = W[1:rn1, :]
#       hssA.W2 = W[rn1+1:end, :]
#     end
#   end
#   return Brow, Bcol
# end

function _compress_block!(A::Matrix{T}; tol, reltol) where T
  B = copy(A)
  Q, R, p = prrqr!(A, tol; reltol)
  rk = min(size(R)...)
  return Q[:,1:rk], R[1:rk, invperm(p)]
end

# ## Recompression algorithm
# function hss_recompress!(hssA::HssMatrix{T}, tol=tol; reltol=reltol) where {T}
#   if hssA.leafnode; return hssA; end
#   # a prerequisite for this algorithm to work is that generators are orthonormal
#   orthonormalize_generators!(hssA)
#   # define Brow, Bcol
#   Brow = Array{T}(undef, 0, 0)
#   Bcol = Array{T}(undef, 0, 0)
#   hss_recompress_rec!(hssA, Brow, Bcol, tol; reltol)
# end

# # recursive definition
# function hss_recompress_rec!(hssA::HssMatrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, tol=tol; reltol=reltol) where {T}
#   if hssA.leafnode; error("recompression called on a leafnode"); end
#   if hssA.rootnode
#     # compress B12, B21 via something that resembles the SVD
#     P1, S2 = compress_block!(hssA.B12, tol; reltol)
#     Q2, T1 = compress_block!(copy(hssA.B12'), tol; reltol)
#     P2, S1 = compress_block!(hssA.B21, tol; reltol)
#     Q1, T2 = compress_block!(copy(hssA.B21'), tol; reltol)

#     hssA.B12 = S2*Q2
#     hssA.B21 = S1*Q1

#     # pass information to children and proceed recursively
#     if !hssA.A11.leafnode
#       hssA.A11.R1 = hssA.A11.R1*P1
#       hssA.A11.R2 = hssA.A11.R2*P1
#       hssA.A11.W1 = hssA.A11.W1*Q1
#       hssA.A11.W2 = hssA.A11.W2*Q1
#     else
#       hssA.A11.U = hssA.A11.U*P1
#       hssA.A11.V = hssA.A11.V*Q1
#     end
#     # repeat for A22
#     if !hssA.A22.leafnode
#       hssA.A22.R1 = hssA.A22.R1*P2
#       hssA.A22.R2 = hssA.A22.R2*P2
#       hssA.A22.W1 = hssA.A22.W1*Q2
#       hssA.A22.W2 = hssA.A22.W2*Q2
#     else
#       hssA.A22.U = hssA.A22.U*P2
#       hssA.A22.V = hssA.A22.V*Q2
#     end

#     # call recompression recursively
#     if !hssA.A11.leafnode
#       hss_recompress_rec!(hssA.A11, hssA.B12, copy(hssA.B21'), tol; reltol)
#     end
#     if !hssA.A22.leafnode
#       hss_recompress_rec!(hssA.A22, hssA.B21, copy(hssA.B12'), tol; reltol)
#     end
#   else
#     # compress B12
#     Brow1 = hcat(hssA.B12, hssA.R1*Brow)
#     Bcol2 = hcat(hssA.B12', hssA.W2*Bcol)
#     P1, S2 = compress_block!(Brow1, tol; reltol)
#     Q2, T1 = compress_block!(Bcol2, tol; reltol)
#     # get the original number of columns in B12
#     rm1, rn2 = size(hssA.B12)
#     hssA.B12 = S2[:,1:rn2]*Q2
#     Brow1 = S2[:, rn2+1:end]
#     Bcol2 = T1[:, rm1+1:end]
#     # compress B21
#     Brow2 = hcat(hssA.B21, hssA.R2*Brow)
#     Bcol1 = hcat(hssA.B21', hssA.W1*Bcol)
#     P2, S1 = compress_block!(Brow2, tol; reltol)
#     Q1, T2 = compress_block!(Bcol1, tol; reltol)
#     # get the original number of columns in B21
#     rm2, rn1 = size(hssA.B21)
#     hssA.B21 = S1[:,1:rn1]*Q1
#     Brow2 = S1[:,rn1+1:end]
#     Bcol1 = T2[:, rm2+1:end]

#     # update generators of A11
#     if !hssA.A11.leafnode
#       hssA.A11.R1 = hssA.A11.R1*P1
#       hssA.A11.R2 = hssA.A11.R2*P1
#       hssA.A11.W1 = hssA.A11.W1*Q1
#       hssA.A11.W2 = hssA.A11.W2*Q1
#     else
#       hssA.A11.U = hssA.A11.U*P1
#       hssA.A11.V = hssA.A11.V*Q1
#     end
#     # update generators in A22
#     if !hssA.A22.leafnode
#       hssA.A22.R1 = hssA.A22.R1*P2
#       hssA.A22.R2 = hssA.A22.R2*P2
#       hssA.A22.W1 = hssA.A22.W1*Q2
#       hssA.A22.W2 = hssA.A22.W2*Q2
#     else
#       hssA.A22.U = hssA.A22.U*P2
#       hssA.A22.V = hssA.A22.V*Q2
#     end

#     # call recompression recursively
#     if !hssA.A11.leafnode
#       Brow1 = hcat(hssA.B12, S2[:,rn2+1:end])
#       Bcol1 = hcat(hssA.B21', T2[:,rm2+1:end])
#       hss_recompress_rec!(hssA.A11, Brow1, Bcol1, tol; reltol)
#     end
#     if !hssA.A22.leafnode
#       Brow2 = hcat(hssA.B21, S1[:,rn1+1:end])
#       Bcol2 = hcat(hssA.B12', T1[:,rm1+1:end])
#       hss_recompress_rec!(hssA.A22, Brow2, Bcol2, tol; reltol)
#     end
#   end
# end

## Randomized compression will go here...