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
  zl = _matvecup(hssA.A11, x[1:n1,:])
  zr = _matvecup(hssA.A22, x[n1+1:end,:])
  z = BinaryNode{Matrix{T}}(hssA.W1' * zl.data + hssA.W2' * zr.data)
  z.left = zl
  z.right = zr
  return z
end

_matvecdown(hssA::HssLeaf{T}, x::Matrix{T}, z::BinaryNode{Matrix{T}}, b::Matrix{T}) where T = hssA.D * x + hssA.U * b;
function _matvecdown(hssA::HssNode{T}, x::Matrix{T}, z::BinaryNode{Matrix{T}}, b::Matrix{T}) where T
  n1 = hssA.sz1[2]
  b1 = hssA.B12 * z.right.data + hssA.R1 * b
  b2 = hssA.B21 * z.left.data + hssA.R2 * b
  y1 = _matvecdown(hssA.A11, x[1:n1,:], z.left, b1)
  y2 = _matvecdown(hssA.A22, x[n1+1:end,:], z.right, b2)
  return [y1; y2]
end
