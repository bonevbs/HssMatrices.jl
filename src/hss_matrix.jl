### Definitions of datastructures and basic constructors and operators
# Written by Boris Bonev, Jan. 2021

## new datastructure which splits the old one into two parts to avoid unnecessary allocations
# definition of leaf nodes
mutable struct HssLeaf{T<:Number} #<: AbstractMatrix{T}
  D ::Matrix{T}
  U ::Matrix{T}
  V ::Matrix{T}

  # constructor
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
    new{T}(A11, A22, B12, B21, size(A11), size(A22), Matrix{Float64}(undef,size(A11,1),0), Matrix{Float64}(undef,size(A11,2),0), Matrix{Float64}(undef,size(A22,1),0), Matrix{Float64}(undef,size(A22,2),0))
  end
  function HssNode(A11::Union{HssLeaf{T}, HssNode{T}}, A22::Union{HssLeaf{T}, HssNode{T}}, B12::Matrix{T}, B21::Matrix{T}, 
    R1::Matrix{T}, W1::Matrix{T}, R2::Matrix{T}, W2::Matrix{T}) where T
    if size(R1,2) != size(R2,2) throw(ArgumentError("R1 and R2 must have same number of columns")) end
    if size(W1,2) != size(W2,2) throw(ArgumentError("W1 and W2 must have same number of rows")) end
    new{T}(A11, A22, B12, B21, size(A11), size(A22), R1, W1, R2, W2)
  end
end

# exterior constructors
#HssNode(A11::Union{HssLeaf, HssNode}, A22::Union{HssLeaf, HssNode}, B12::Matrix, B21::Matrix, ::Nothing, ::Nothing, ::Nothing, ::Nothing) = HssNode(A11, A22, B12, B21)
# TODO: add constructors that use compression methods

# convenience alias (maybe unnecessary)
const HssMatrix{T} = Union{HssLeaf{T}, HssNode{T}}


## Base overrides
Base.eltype(::Type{HssLeaf{T}}) where T = T
Base.eltype(::Type{HssNode{T}}) where T = T

Base.size(hssA::HssLeaf) = size(hssA.D)
Base.size(hssA::HssNode) = hssA.sz1 .+ hssA.sz2
Base.size(hssA::HssMatrix, dim::Integer) = size(hssA)[dim]

Base.show(io::IO, hssA::HssLeaf) = print(io, "$(size(hssA)) HssLeaf{$(eltype(hssA))}")
Base.show(io::IO, hssA::HssNode) = print(io, "$(size(hssA)) HssNode{$(eltype(hssA))}")

Base.copy(hssA::HssLeaf) = HssLeaf{eltype(hssA)}(copy(hssA.D), copy(hssA.U), copy(hssA.V))
Base.copy(hssA::HssNode) = HssNode{eltype(hssA)}(copy(hssA.A11), copy(hssA.A22), copy(hssA.B12), copy(hssA.B21), copy(R1), copy(W1), copy(R2), copy(W2))

# ## conversion
# #convert(::HssMatrix{T}, hssA::HssMatrix) where {T} = HssMatrix()

# ## Typecasting routines
# # function HssMatrix{T<:Number}(hssA::HssMatrix{S}) where S
# #   isdefined() ? HssMatrix
# # end

## HSS specific routines
hssrank(hssA::HssLeaf) = 0
hssrank(hssA::HssNode) = max(hssrank(hssA.A11), hssrank(hssA.A22), rank(hssA.B12), rank(hssA.B21))


# # construct full matrix from HSS
# function Base.Matrix(hssA::HssMatrix{T}) where {T}
#   n = size(hssA,2)
#   return hssA * Union{Matrix{T}, Nothing}(I, n, n)
# end

# # alternatively we can form the full matrix in a more straight-forward fashion