### Compute the ULV factorization of a HSS matrix
#
# as seen in
# Chandrasekaran, S., Gu, M., & Pals, T. (2006). A Fast $ULV$ Decomposition Solver for Hierarchically Semiseparable Representations.
# SIAM Journal on Matrix Analysis and Applications, 28(3), 603–622. https://doi.org/10.1137/S0895479803436652
#
# Written by Boris Bonev, Nov. 2020

# load efficient BLAS and LAPACK routines for factorizations
import LinearAlgebra.LAPACK.geqlf!
import LinearAlgebra.LAPACK.gelqf!
import LinearAlgebra.LAPACK.ormql!
import LinearAlgebra.LAPACK.ormlq!
import LinearAlgebra.BLAS.trsm

## function for direct solution using the implicit ULV factorization
ulvfactsolve(hssA::HssLeaf{T}, b::Matrix{T}) where T = hssA.D\b
function ulvfactsolve(hssA::HssNode{T}, b::Matrix{T}) where T
  z = zeros(size(hssA,2), size(b,2))
  _, _, _, _, _, _, _, QV = _ulvfactsolve!(hssA, b, z, 0; rootnode=true)
  z = _ulvsolve_topdown!(QV, z)
  return z
end

# core routine to reduce the rows and triangularize the diagonal block via QR/LQ decompositions and immediately apply them to b
function _ulvreduce!(D::Matrix{T}, U::Matrix{T}, V::Matrix{T}, b::Matrix{T}) where T
  T <: Complex ? adj = 'C' : adj = 'T'
  m, n = size(D)
  k = size(U, 2)
  nk = min(m-k,n)
  ind = 1:m-k
  cind = m-k+1:m
  # can't be compressed, exit early
  if k >= m
    u = zeros(m, size(b,2))
    zloc = Matrix{T}(undex, 0, size(b,2))
  else
    # form QL decomposition of the row generators and apply it
    qlf = geqlf!(U);
    U = tril(U[end-k+1:end,:]) # k x k block
    ormql!('L', adj, qlf..., D) # transform the diagonal block
    ormql!('L', adj, qlf..., b) # transform the right-hand side
    # Form the LQ decomposition of the first m-k rows of D
    lqf = gelqf!(D[1:end-k,:])
    L1 = tril(lqf[1])
    L1 = L1[:,1:nk]
    L2 = ormlq!('R', adj, lqf..., D[end-k+1:end,:]) # update the bottom block of the diagonal block
    zloc = trsm('L', 'L', 'N', 'N', 1., L1, b[ind,:])
    b = b[cind, :] .- L2[:,1:nk] * zloc # remove contribution in the uncompressed parts
    V = ormlq!('L', 'N', lqf..., V) # compute the updated off-diagonal generators on the right
    u = V[ind,:]' * zloc # compute update vector to be passed on
    # pass on uncompressed parts of the problem
    D = L2[:, nk+1:end]
    V = V[nk+1:end, :]
  end
  return D, U, V, b, zloc, u, m-k, nk, lqf
end

# tree traversal that handles the hierarchical ULV factorization
function _ulvfactsolve!(hssA::HssLeaf{T}, b::Matrix{T}, z::Matrix{T}, co::Int; rootnode=false) where T
  cols = collect(co .+ (1:size(hssA,2)))
  D = copy(hssA.D); U = copy(hssA.U); V = copy(hssA.V)
  D, U, V, b, zloc, u, mk, nk, lqf =_ulvreduce!(D, U, V, b)
  z[cols[1:mk], :] = zloc
  QV = BinaryNode((cols, lqf...))
  return b, u, D, U, V, cols, nk, QV 
end
function _ulvfactsolve!(hssA::HssNode{T}, b::Matrix{T}, z::Matrix{T}, co::Int; rootnode=false) where T
  m1, n1 = hssA.sz1; m2, n2 = hssA.sz2
  b1 = b[1:m1, :]; b2 = b[m1+1:end, :] # performance could be further improved by getting rid of those allocations
  b1, u1, D1, U1, V1, cols1, nk1, QV1 = _ulvfactsolve!(hssA.A11, b1, z, co)
  b2, u2, D2, U2, V2, cols2, nk2, QV2 = _ulvfactsolve!(hssA.A22, b2, z, co+n1)

  # merge nodes to form new diagonal block 
  b = [b1; b2] .- [U1*hssA.B12*u2; U2*hssA.B21*u1]
  D = [D1 U1*hssA.B12*V2'; U2*hssA.B21*V1' D2]
  cols = [cols1[nk1+1:end]; cols2[nk2+1:end]] # to re-adjust local numbering

  U = [U1*hssA.R1; U2*hssA.R2]
  V = [V1*hssA.W1; V2*hssA.W2]

  # now if this is the rootnode we should directly solve?
  if rootnode
    z[cols, :] = D\b
    u = Matrix{T}(undef,0,size(b,2))
    mk, nk = size(D)
    QV = BinaryNode{Tuple{Vector{Int}, Matrix{T}, Vector{T}}}()
  else
    D, U, V, b, zloc, u, mk, nk, lqf = _ulvreduce!(D, U, V, b)
    u = u .+ hssA.W1'*u1 .+ hssA.W2'*u2
    z[cols[1:mk], :] = zloc
    QV = BinaryNode((cols, lqf...))
  end
  QV.left = QV1; QV.right = QV2
  return b, u, D, U, V, cols, nk, QV 
end

function _ulvsolve_topdown!(QV::BinaryNode{Tuple{Vector{Int}, Matrix{T}, Vector{T}}}, z::Matrix{T}) where T
  T <: Complex ? adj = 'C' : adj = 'T'
  if isdefined(QV, :data)
    (cols, Q, tau) = QV.data
    z[cols, :] = ormlq!('L', adj, Q, tau, z[cols, :])
  end
  if !isnothing(QV.left) z = _ulvsolve_topdown!(QV.left, z) end
  if !isnothing(QV.right) z = _ulvsolve_topdown!(QV.right, z) end
  return z
end

# temporary name for function that actually just computes the ULV factorization
function hssdivide(hssA::HssMatrix{T}, hssB::HssMatrix{T}) where T
  outputs = _hssdivide(hssA)

end

# core routine to reduce the rows and triangularize the diagonal block via QR/LQ decompositions
function _ulvreduce!(D::Matrix{T}, U::Matrix{T}, V::Matrix{T}) where T
  T <: Complex ? adj = 'C' : adj = 'T'
  m, n = size(D)
  k = size(U, 2)
  nk = min(m-k,n)
  ind = 1:m-k
  cind = m-k+1:m
  # can't be compressed, exit early
  if k >= m
    u = zeros(m, size(b,2))
    zloc = Matrix{T}(undex, 0, size(b,2))
  else
    # form QL decomposition of the row generators and apply it
    qlf = geqlf!(U);
    U = tril(U[end-k+1:end,:]) # k x k block
    ormql!('L', adj, qlf..., D) # transform the diagonal block
    ormql!('L', adj, qlf..., b) # transform the right-hand side
    # Form the LQ decomposition of the first m-k rows of D
    lqf = gelqf!(D[1:end-k,:])
    L1 = tril(lqf[1])
    L1 = L1[:,1:nk]
    L2 = ormlq!('R', adj, lqf..., D[end-k+1:end,:]) # update the bottom block of the diagonal block
    zloc = trsm('L', 'L', 'N', 'N', 1., L1, b[ind,:])
    b = b[cind, :] .- L2[:,1:nk] * zloc # remove contribution in the uncompressed parts
    V = ormlq!('L', 'N', lqf..., V) # compute the updated off-diagonal generators on the right
    u = V[ind,:]' * zloc # compute update vector to be passed on
    # pass on uncompressed parts of the problem
    D = L2[:, nk+1:end]
    V = V[nk+1:end, :]
  end
  return D, U, V, L1, qlf, lqf # , u, m-k, nk
end

# recursive function that operates only on the leaves
function _ulvfact(hssA::HssLeaf{T}, co::Int; rootnode=false) where T
  cols = collect(co .+ (1:size(hssA,2)))
  D = copy(hssA.D); U = copy(hssA.U); V = copy(hssA.V)
  D, U, V, L1, qlf, lqf, mk, nk =_ulvreduce!(D, U, V)
  QU = BinaryNode((cols, qlf...))
  QL = BinaryNode((cols, L1))
  QV = BinaryNode((cols, lqf...))
end
function _ulvfact(hssA::HssNode{T}, co::Int; rootnode=false) where T
  m1, n1 = hssA.sz1; m2, n2 = hssA.sz2
  b1 = b[1:m1, :]; b2 = b[m1+1:end, :] # performance could be further improved by getting rid of those allocations
  b1, u1, D1, U1, V1, cols1, nk1, QV1 = _ulvfact(hssA.A11, b1, z, co)
  b2, u2, D2, U2, V2, cols2, nk2, QV2 = _ulvfact(hssA.A22, b2, z, co+n1)
end