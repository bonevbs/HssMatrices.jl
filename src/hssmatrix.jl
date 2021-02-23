### Definitions of datastructures and basic constructors and operators
# Written by Boris Bonev, Jan. 2021

## new datastructure which splits the old one into two parts to avoid unnecessary allocations
# definition of leaf nodes
mutable struct HssLeaf{T<:Number} #<: AbstractMatrix{T}
  D ::Matrix{T}
  U ::Matrix{T}
  V ::Matrix{T}

  # constructor
  function HssLeaf(D::Matrix{T}) where T
    m, n = size(D)
    new{T}(D, Matrix{T}(undef, m, 0), Matrix{T}(undef, n, 0))
  end
  function HssLeaf(D::Matrix{T}, U::Matrix{T}, V::Matrix{T}) where T
    if size(D,1) != size(U,1) throw(ArgumentError("D and U must have same number of rows")) end
    if size(D,2) != size(V,1) throw(ArgumentError("D and V must have same number of columns")) end
    new{T}(D, U, V)
  end
end

# definition of branch nodes
mutable struct HssNode{T<:Number} #<: AbstractMatrix{T}
  A11 ::Union{HssNode{T}, HssLeaf{T}}
  A22 ::Union{HssNode{T}, HssLeaf{T}}
  B12 ::Matrix{T}
  B21 ::Matrix{T}

  sz1 ::Tuple{Int, Int}
  sz2 ::Tuple{Int, Int}

  R1 ::Matrix{T}
  W1 ::Matrix{T}
  R2 ::Matrix{T}
  W2 ::Matrix{T}

  # internal constructors with checks for dimensions
  function HssNode(A11::Union{HssLeaf{T}, HssNode{T}}, A22::Union{HssLeaf{T}, HssNode{T}}, B12::Matrix{T}, B21::Matrix{T}) where T
    kr1, kw1 = gensize(A11); kr2, kw2 = gensize(A22)
    new{T}(A11, A22, B12, B21, size(A11), size(A22),
      Matrix{Float64}(undef,kr1,0), Matrix{Float64}(undef,kw1,0), Matrix{Float64}(undef,kr2,0), Matrix{Float64}(undef,kw2,0))
  end
  function HssNode(A11::Union{HssLeaf{T}, HssNode{T}}, A22::Union{HssLeaf{T}, HssNode{T}}, B12::Matrix{T}, B21::Matrix{T}, 
    R1::Matrix{T}, W1::Matrix{T}, R2::Matrix{T}, W2::Matrix{T}) where T
    if size(R1,2) != size(R2,2) throw(DimensionMismatch("R1 and R2 must have same number of columns")) end
    if size(W1,2) != size(W2,2) throw(DimensionMismatch("W1 and W2 must have same number of rows")) end
    new{T}(A11, A22, B12, B21, size(A11), size(A22), R1, W1, R2, W2)
  end
end

# exterior constructors
#HssNode(A11::Union{HssLeaf, HssNode}, A22::Union{HssLeaf, HssNode}, B12::Matrix, B21::Matrix, ::Nothing, ::Nothing, ::Nothing, ::Nothing) = HssNode(A11, A22, B12, B21)
# TODO: add constructors that use compression methods

# convenience alias (maybe unnecessary)
const HssMatrix{T} = Union{HssLeaf{T}, HssNode{T}}
@inline isleaf(hssA::HssMatrix) = typeof(hssA) <: HssLeaf # check whether making this inline speeds up things ?
@inline isbranch(hssA::HssMatrix) = typeof(hssA) <: HssNode

## Base overrides
eltype(::Type{HssLeaf{T}}) where T = T
eltype(::Type{HssNode{T}}) where T = T

size(hssA::HssLeaf) = size(hssA.D)
size(hssA::HssNode) = hssA.sz1 .+ hssA.sz2
size(hssA::HssMatrix, dim::Integer) = size(hssA)[dim]

show(io::IO, hssA::HssLeaf) = print(io, "$(size(hssA,1))x$(size(hssA,2)) HssLeaf{$(eltype(hssA))}")
show(io::IO, hssA::HssNode) = print(io, "$(size(hssA,1))x$(size(hssA,2)) HssNode{$(eltype(hssA))}")

copy(hssA::HssLeaf) = HssLeaf(copy(hssA.D), copy(hssA.U), copy(hssA.V))
copy(hssA::HssNode) = HssNode(copy(hssA.A11), copy(hssA.A22), copy(hssA.B12), copy(hssA.B21), copy(hssA.R1), copy(hssA.W1), copy(hssA.R2), copy(hssA.W2))

# maybe replace that later wit ha lazy adjjoint, which swaps cols and rows when called
# TODO: create AbstractHssMatrix type to contain the adjoint as well
# _getproperty(hssA::Adjoint{T, HssLeaf{T}}, ::Val{:D}l) where T  = adjoint(hssA.D)
# _getproperty(hssA::Adjoint{T, HssLeaf{T}}, ::Val{:U}l) where T  = hssA.V
# _getproperty(hssA::Adjoint{T, HssLeaf{T}}, ::Val{:V}l) where T  = hssA.U
# _getproperty(hssA::Adjoint{T, HssNode{T}}, ::Val{:A11}l) where T  = adjoint(hssA.A11)
# Base.getproperty(hssA::Adjoint{T, HssMatrix{T}}, s::Symbol) where T = _getproperty(hssA, Val{s}())
adjoint(hssA::HssLeaf) = HssLeaf(collect(hssA.D'), hssA.V, hssA.U)
adjoint(hssA::HssNode) = HssNode(adjoint(hssA.A11), adjoint(hssA.A22), collect(hssA.B21'), collect(hssA.B12'), hssA.W1, hssA.R1, hssA.W2, hssA.R2)
transpose(hssA::HssLeaf) = HssLeaf(collect(transpose(hssA.D)), hssA.V, hssA.U)
transpose(hssA::HssNode) = HssNode(transpose(hssA.A11), transpose(hssA.A22), collect(transpose(hssA.B21)), collect(transpose(hssA.B12)), hssA.W1, hssA.R1, hssA.W2, hssA.R2)

# Define Matlab-like convenience functions, which are used throughout the library
blkdiag(A::Matrix, B::Matrix) = [A zeros(size(A,1), size(B,2)); zeros(size(B,1), size(A,2)) B]
blkdiag(A::Matrix... ) = blkdiag(A[1], blkdiag(A[2:end]...))

## basic algebraic operations (taken and modified from LowRankApprox.jl)
for op in (:+,:-)
  @eval begin
    $op(hssA::HssLeaf) = HssLeaf($op(hssA.D), hssA.U, hssA.V)
    $op(hssA::HssNode) = HssNode($op(hssA.A11), $op(hssA.A22), $op(hssA.B12), $op(hssA.B21), hssA.R1, hssA.W1, hssA.R2, hssA.W2)

    $op(a::Bool, hssA::HssMatrix{Bool}) = error("Not callable")
    $op(L::HssMatrix{Bool}, a::Bool) = error("Not callable")
    #$op(a::Number, hssA::HssMatrix) = $op(LowRankMatrix(Fill(a,size(L))), L)
    #$op(L::HssMatrix, a::Number) = $op(L, LowRankMatrix(Fill(a,size(L))))

    function $op(hssA::HssLeaf, hssB::HssLeaf)
      size(hssA) == size(hssB) || throw(DimensionMismatch("A has dimensions $(size(hssA)) but B has dimensions $(size(hssB))"))
      HssLeaf($op(hssA.D, hssB.D), [hssA.U hssB.U], [hssA.V hssB.V])
    end
    function $op(hssA::HssNode, hssB::HssNode)
      hssA.sz1 == hssB.sz1 || throw(DimensionMismatch("A11 has dimensions $(hssA.sz1) but B11 has dimensions $(hssB.sz1)"))
      hssA.sz2 == hssB.sz2 || throw(DimensionMismatch("A22 has dimensions $(hssA.sz2) but B22 has dimensions $(hssA.sz2)"))
      hssC = HssNode($op(hssA.A11, hssB.A11), $op(hssA.A22, hssB.A22), blkdiag(hssA.B12, $op(hssB.B12)), blkdiag(hssA.B21, $op(hssB.B21)),
        blkdiag(hssA.R1, hssB.R1), blkdiag(hssA.W1, hssB.W1), blkdiag(hssA.R2, hssB.R2), blkdiag(hssA.W2, hssB.W2))
    end
    #$op(L::LowRankMatrix,A::Matrix) = $op(promote(L,A)...)
    #$op(A::Matrix,L::LowRankMatrix) = $op(promote(A,L)...)
  end
end

# # getindex routines
# # _getindex routines assume sorted arrays
# _getindex(hssA::HssLeaf, i, j) = hssA.D[i, j], U[i, :], V[j, :]
# function _getindex(hssA::HssNode, i, j)

# end

# matrix division involving HSS matrices
\(hssA::HssMatrix, B::Matrix) = ulvfactsolve(hssA, B)
\(hssA::HssMatrix, hssB::HssMatrix) = ldiv!(hssA, copy(hssB))

# Scalar multiplication
*(a::Number, hssA::HssLeaf) = HssLeaf(a*hssA.D, hssA.U, hssA.V)
*(a::Number, hssA::HssNode) = HssNode(a*hssA.A11, a*hssA.A22, a*hssA.B12, a*hssA.B21, hssA.R1, hssA.W1, hssA.R2, hssA.W2)
*(hssA::HssLeaf, a::Number) = *(a, hssA)
*(hssA::HssNode, a::Number) = *(a, hssA)

## Some more fundamental operations
# compute the HSS rank
hssrank(hssA::HssLeaf) = 0
hssrank(hssA::HssNode) = max(hssrank(hssA.A11), hssrank(hssA.A22), rank(hssA.B12), rank(hssA.B21))
gensize(hssA::HssLeaf) = size(hssA.U,2), size(hssA.V,2)
function gensize(hssA::HssNode)
  (kr = size(hssA.R1,2)) == size(hssA.R2,2) || throw(DimensionMismatch("dimensions of column-translators do not match"))
  (kw = size(hssA.W1,2)) == size(hssA.W2,2) || throw(DimensionMismatch("dimensions of row-translators do not match"))
  return kr, kw
end

# return a full matrix
# TODO: change this into a convert routine
full(hssA::HssMatrix) = _full(hssA)[1]
_full(hssA::HssLeaf) = hssA.D, hssA.U, hssA.V
function _full(hssA::HssNode)
  A11, U1, V1 = _full(hssA.A11)
  A22, U2, V2 = _full(hssA.A22)
  return [A11 U1*hssA.B12*V2'; U2*hssA.B21*V1' A22], [U1*hssA.R1; U2*hssA.R2], [V1*hssA.W1; V2*hssA.W2]
end

# useful routine to check whether dimensions are compatible
checkdims(hssA::HssMatrix)= _checkdims(hssA, 1)[1]
function _checkdims(hssA::HssLeaf, i::Int)
  compatible = (size(hssA.D,1) == size(hssA.U,1)) && (size(hssA.D,2) == size(hssA.V,1))
  if !compatible println("dimensions don't match in node ", i) end
  return compatible, i+1
end
function _checkdims(hssA::HssNode, i::Int)
  comp1, i = _checkdims(hssA.A11, i)
  comp2, i = _checkdims(hssA.A22, i)
  r1, w1 = gensize(hssA.A11); r2, w2 = gensize(hssA.A22)
  compatible = (r1 == size(hssA.R1,1)) && (r2 == size(hssA.R2,1)) && (w1 == size(hssA.W1,1)) && (w2 == size(hssA.W2,1))
  if !compatible println("dimensions don't match in node ", i) end
  return compatible && comp1 && comp2, i+1
end

# remove leaves on the bottom level
prune_leaves!(hssA::HssLeaf) = hssA
function prune_leaves!(hssA::HssNode)
  if isleaf(hssA.A11) && isleaf(hssA.A22)
    return HssLeaf(_full(hssA)...)
  else
    hssA.A11 = prune_leaves!(hssA.A11)
    hssA.A22 = prune_leaves!(hssA.A22)
    return hssA
  end
end