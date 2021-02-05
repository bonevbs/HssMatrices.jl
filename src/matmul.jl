### Defines all multiplication routines for HssMatrices
#
# Matrix-vector multiplication
# as seen in
# Chandrasekaran, S., Dewilde, P., Gu, M., Lyons, W., & Pals, T. (2006). A fast solver for HSS representations via sparse matrices.
# SIAM Journal on Matrix Analysis and Applications, 29(1), 67–81. https://doi.org/10.1137/050639028
#
# Written by Boris Bonev, Nov. 2020

## PROMOTOE - look at LowRankApprox again
## TODO: read about promotion and improve the code

# not sure this is needed
if !isdefined(@__MODULE__, :BinaryNode)
  include("binarytree.jl")
end

## Multiplication with vectors/matrices
*(hssA::HssLeaf, x::Matrix) = hssA.D * x
function *(hssA::HssNode, x::Matrix)
  if size(hssA,2) != size(x,1); error("dimensions do not match"); end
  z = _matvecup(hssA, x) # saves intermediate steps of multiplication in a binary tree structure
  b = Matrix{eltype(x)}(undef,0,size(x,2))
  return _matvecdown(hssA, x, z, b)
end

*(hssA::HssMatrix, x::Vector) = hssA * reshape(x, length(x), 1)

## auxiliary functions for the fast multiplication algorithm
# post-ordered step of mat-vec
_matvecup(hssA::HssLeaf{T}, x::Matrix{T}) where T = BinaryNode{Matrix{T}}(hssA.V' * x)
function _matvecup(hssA::HssNode{T}, x::Matrix{T}) where T
  n1 = hssA.sz1[2]
  z1 = _matvecup(hssA.A11, x[1:n1,:])
  z2 = _matvecup(hssA.A22, x[n1+1:end,:])
  z = BinaryNode{Matrix{T}}(hssA.W1' * z1.data + hssA.W2' * z2.data, z1, z2)
  return z
end

_matvecdown(hssA::HssLeaf{T}, x::Matrix{T}, z::BinaryNode{Matrix{T}}, b::Matrix{T}) where T = hssA.D * x + hssA.U * b
function _matvecdown(hssA::HssNode{T}, x::Matrix{T}, z::BinaryNode{Matrix{T}}, b::Matrix{T}) where T
  n1 = hssA.sz1[2]
  b1 = hssA.B12 * z.right.data + hssA.R1 * b
  b2 = hssA.B21 * z.left.data + hssA.R2 * b
  y1 = _matvecdown(hssA.A11, x[1:n1,:], z.left, b1)
  y2 = _matvecdown(hssA.A22, x[n1+1:end,:], z.right, b2)
  return [y1; y2]
end

## multiplication of two HSS matrices
*(hssA::HssLeaf, hssB::HssLeaf) = HssLeaf(hssA.D*hssB.D, hssA.U, hssB.V)
function *(hssA::HssNode, hssB::HssNode)
  # implememnt cluster equality checks
  #if cluster(hssA,2) != cluster(hssB,1); throw(DimensionMismatch("clusters of hssA and hssB must be matching")) end
  Z = _matmatup(hssA, hssB) # saves intermediate steps of multiplication in a binary tree structure
  F1 = hssA.B12 * Z.right.data * hssB.B21
  F2 = hssA.B21 * Z.left.data * hssB.B12
  B12 = blkdiag(hssA.B12, hssB.B12)
  B21 = blkdiag(hssA.B21, hssB.B21)
  A11 = _matmatdown(hssA.A11, hssB.A11, Z.left, F1)
  A22 = _matmatdown(hssA.A22, hssB.A22, Z.right, F2)
  hssC = HssNode(A11, A22, B12, B21)
  return hssC
end

_matmatup(hssA::HssLeaf{T}, hssB::HssLeaf{T}) where T = BinaryNode{Matrix{T}}(hssA.V' * hssB.U)
function _matmatup(hssA::HssNode{T}, hssB::HssNode{T}) where T
  Z1 = _matmatup(hssA.A11, hssB.A11)
  Z2 = _matmatup(hssA.A22, hssB.A22)
  return BinaryNode(hssA.W1' * Z1.data * hssB.R1 + hssA.W2' * Z2.data * hssB.R2, Z1, Z2)
end

function _matmatdown(hssA::HssLeaf{T}, hssB::HssLeaf{T}, Z::BinaryNode{Matrix{T}}, F::Matrix{T}) where T
  D = hssA.D * hssB.D + hssA.U * F * hssB.V'
  U = [hssA.U hssA.D * hssB.U]
  V = [hssB.D' * hssA.V hssB.V]
  return HssLeaf(D, U, V)
end
function _matmatdown(hssA::HssNode{T}, hssB::HssNode{T}, Z::BinaryNode{Matrix{T}}, F::Matrix{T}) where T
  # evaluate cross terms
  F1 = hssA.B12 * Z.right.data * hssB.B21 + hssA.R1 * F * hssB.W1'
  F2 = hssA.B21 * Z.left.data * hssB.B12 + hssA.R2 * F * hssB.W2'
  B12 = [hssA.B12 hssA.R1 * F * hssB.W2'; zeros(size(hssB.B12, 1), size(hssA.B12,2)) hssB.B12]
  B21 = [hssA.B21 hssA.R2 * F * hssB.W1'; zeros(size(hssB.B21, 1), size(hssA.B21,2)) hssB.B21]
  R1 = [hssA.R1 hssA.B12 * Z.right.data * hssB.R2; zeros(size(hssB.R1,1), size(hssA.R1,2)) hssB.R1];
  W1 = [hssA.W1 zeros(size(hssA.W1,1), size(hssB.W1,2)); hssB.B21' * Z.right.data' * hssA.W2 hssB.W1];
  R2 = [hssA.R2 hssA.B21 * Z.left.data * hssB.R1; zeros(size(hssB.R2,1), size(hssA.R2,2)) hssB.R2];
  W2 = [hssA.W2 zeros(size(hssA.W2,1), size(hssB.W2,2)); hssB.B12' * Z.left.data' * hssA.W1 hssB.W2];
  A11 = _matmatdown(hssA.A11, hssB.A11, Z.left, F1)
  A22 = _matmatdown(hssA.A22, hssB.A22, Z.right, F2)
  return HssNode(A11, A22, B12, B21, R1, W1, R2, W2)
end