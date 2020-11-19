### Defines all multiplication routines for HssMatrices

## Matrix-vector multiplication
# as seen in
# Chandrasekaran, S., Dewilde, P., Gu, M., Lyons, W., & Pals, T. (2006). A fast solver for HSS representations via sparse matrices.
# SIAM Journal on Matrix Analysis and Applications, 29(1), 67–81. https://doi.org/10.1137/050639028

include("./cluster_trees.jl")

function *(hssA::HssMatrix{T}, x::Matrix{S}) where {T<:Number, S<:Number}
  if !hssA.leafnode
    if size(hssA,2) != size(x,1); error("dimensions do not match"); end
    t = hss_matvec_bottomup(hssA, x) # saves intermediate steps of multiplication in a binary tree structure
    b = Matrix{eltype(x)}(undef,0,0)
    y = hss_matvec_topdown(hssA, x, t, b)
  else
    y = hssA.D * x
  end
  return y
end

# post-ordered step of mat-vec
function hss_matvec_bottomup(hssA::HssMatrix{T}, x::Matrix{S}) where {T<:Number, S<:Number}
  xt = BinaryNode(Matrix{S}(undef,0,0))
  if !hssA.leafnode
    xt.left = hss_matvec_bottomup(hssA.A11, x[1:hssA.n1,:])
    xt.right = hss_matvec_bottomup(hssA.A22, x[hssA.n1+1:end,:])
    if !hssA.rootnode
      xt.data = hssA.W1' * xt.left.data + hssA.W2' * xt.right.data
    end
  else
    xt.data = hssA.V' * x
  end
  return xt
end

# pre-ordered step of mat-vec
function hss_matvec_topdown(hssA::HssMatrix{T}, x::Matrix{S}, xt::BinaryNode{Matrix{S}}, b::Matrix{S}) where {T<:Number, S<:Number}
  if !hssA.leafnode
    if hssA.rootnode
      b1 = hssA.B12 * xt.right.data
      b2 = hssA.B21 * xt.left.data
    else
      b1 = hssA.B12 * xt.right.data + hssA.R1 * b
      b2 = hssA.B21 * xt.left.data + hssA.R2 * b
    end
    y1 = hss_matvec_topdown(hssA.A11, x[1:hssA.n1,:], xt.left, b1)
    y2 = hss_matvec_topdown(hssA.A22, x[hssA.n1+1:end,:], xt.right, b2)
    y = hcat(y1, y2)
  else
    y = hssA.D * y + hssA.U * f;
  end
  return y
end
