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
# convenience access - TODO: maybe move to hssmatrix.jl in the future
*(hssA::HssMatrix, B::AbstractMatrix) = mul!(similar(B, size(hssA,1), size(B,2)), hssA, B, 1., 0.)
*(A::AbstractMatrix, hssB::HssMatrix) = copy(mul!(similar(A, size(hssB,2), size(A,1)), hssB', copy(A'), 1., 0.)')
*(hssA::HssMatrix, x::AbstractVector) = reshape(hssA * reshape(x, length(x), 1), length(x))

# implement low-level mul! routines
function mul!(C::AbstractMatrix, hssA::HssMatrix, B::AbstractMatrix, α::Number, β::Number)
  size(hssA,2) == size(B,1) ||  throw(DimensionMismatch("First dimension of B does not match second dimension of A. Expected $(size(A, 2)), got $(size(B, 1))"))
  size(C) == (size(hssA,1), size(B,2)) ||  throw(DimensionMismatch("Dimensions of C don't match up with A and B."))
  if isleaf(hssA)
    return mul!(C, hssA.D, B, α, β)
  else
    hssA = rooted(hssA) # this is to avoid top-generators getting in the way if this is called on a sub-block of the HSS matrix
    Z = _matmatup(hssA, B) # saves intermediate steps of multiplication in a binary tree structure
    return _matmatdown!(C, hssA, B, Z, nothing, α, β)
  end
end

## auxiliary functions for the fast multiplication algorithm
# post-ordered step of mat-vec
function _matmatup(hssA::HssMatrix, B::AbstractMatrix)
  if isleaf(hssA)
    return BinaryNode(hssA.V' * B)
  else
    n1 = hssA.sz1[2]
    Z1 = _matmatup(hssA.A11, B[1:n1,:])
    Z2 = _matmatup(hssA.A22, B[n1+1:end,:])
    Z = BinaryNode(hssA.W1'*Z1.data .+ hssA.W2'*Z2.data, Z1, Z2)
    return Z
  end
end

function _matmatdown!(C::AbstractMatrix{T}, hssA::HssMatrix{T}, B::AbstractMatrix{T}, Z::BinaryNode{AT}, F::Union{AbstractMatrix{T}, Nothing}, α::Number, β::Number) where {T, AT<:AbstractArray{T}}
  if isleaf(hssA)
    mul!(C, hssA.D, B , α, β)
    if !isnothing(F); mul!(C, hssA.U, F , α, 1.); end
    return C
  else
    m1, n1 = hssA.sz1
    if !isnothing(F)
      F1 = hssA.B12 * Z.right.data + hssA.R1 * F
      F2 = hssA.B21 * Z.left.data + hssA.R2 * F
    else
      F1 = hssA.B12 * Z.right.data
      F2 = hssA.B21 * Z.left.data
    end
    _matmatdown!(@view(C[1:m1,:]), hssA.A11, @view(B[1:n1,:]), Z.left, F1, α, β)
    _matmatdown!(@view(C[m1+1:end,:]), hssA.A22, @view(B[n1+1:end,:]), Z.right, F2, α, β)
    return C
  end
end

## multiplication of two HSS matrices
function *(hssA::HssMatrix, hssB::HssMatrix)
  if isleaf(hssA) && isleaf(hssA)
    hssC = HssMatrix(hssA.D*hssB.D, hssA.U, hssB.V, true)
  elseif isbranch(hssA) && isbranch(hssA)
    hssA = rooted(hssA); hssB = rooted(hssB)
    # implememnt cluster equality checks
    #if cluster(hssA,2) != cluster(hssB,1); throw(DimensionMismatch("clusters of hssA and hssB must be matching")) end
    Z = _matmatup(hssA, hssB) # saves intermediate steps of multiplication in a binary tree structure
    F1 = hssA.B12 * Z.right.data * hssB.B21
    F2 = hssA.B21 * Z.left.data * hssB.B12
    B12 = blkdiagm(hssA.B12, hssB.B12)
    B21 = blkdiagm(hssA.B21, hssB.B21)
    A11 = _matmatdown!(hssA.A11, hssB.A11, Z.left, F1)
    A22 = _matmatdown!(hssA.A22, hssB.A22, Z.right, F2)
    hssC = HssMatrix(A11, A22, B12, B21, true)
  else
    error("Clusters don't seem to match")
  end
  return hssC
end

function _matmatup(hssA::HssMatrix{T}, hssB::HssMatrix{T}) where T<:Number
  if isleaf(hssA) & isleaf(hssB)
    return BinaryNode{Matrix{T}}(hssA.V' * hssB.U)
  elseif isbranch(hssA) && isbranch(hssB)
    Z1 = _matmatup(hssA.A11, hssB.A11)
    Z2 = _matmatup(hssA.A22, hssB.A22)
    return BinaryNode(hssA.W1' * Z1.data * hssB.R1 + hssA.W2' * Z2.data * hssB.R2, Z1, Z2)
  end
end

function _matmatdown!(hssA::HssMatrix{T}, hssB::HssMatrix{T}, Z::BinaryNode{Matrix{T}}, F::Matrix{T}) where T
  if isleaf(hssA) & isleaf(hssB)
    D = hssA.D * hssB.D + hssA.U * F * hssB.V'
    U = [hssA.U hssA.D * hssB.U]
    V = [hssB.D' * hssA.V hssB.V]
    return HssMatrix(D, U, V, false)
  elseif isbranch(hssA) && isbranch(hssB)
    # evaluate cross terms
    F1 = hssA.B12 * Z.right.data * hssB.B21 + hssA.R1 * F * hssB.W1'
    F2 = hssA.B21 * Z.left.data * hssB.B12 + hssA.R2 * F * hssB.W2'
    B12 = [hssA.B12 hssA.R1 * F * hssB.W2'; zeros(size(hssB.B12, 1), size(hssA.B12,2)) hssB.B12]
    B21 = [hssA.B21 hssA.R2 * F * hssB.W1'; zeros(size(hssB.B21, 1), size(hssA.B21,2)) hssB.B21]
    R1 = [hssA.R1 hssA.B12 * Z.right.data * hssB.R2; zeros(size(hssB.R1,1), size(hssA.R1,2)) hssB.R1];
    W1 = [hssA.W1 zeros(size(hssA.W1,1), size(hssB.W1,2)); hssB.B21' * Z.right.data' * hssA.W2 hssB.W1];
    R2 = [hssA.R2 hssA.B21 * Z.left.data * hssB.R1; zeros(size(hssB.R2,1), size(hssA.R2,2)) hssB.R2];
    W2 = [hssA.W2 zeros(size(hssA.W2,1), size(hssB.W2,2)); hssB.B12' * Z.left.data' * hssA.W1 hssB.W2];
    A11 = _matmatdown!(hssA.A11, hssB.A11, Z.left, F1)
    A22 = _matmatdown!(hssA.A22, hssB.A22, Z.right, F2)
    return HssMatrix(A11, A22, B12, B21, R1, W1, R2, W2, false)
  else
    error("Clusters don't seem to match")
  end
end