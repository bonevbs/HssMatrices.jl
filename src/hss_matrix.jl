### Definitions of datastructures and basic constructors and operators
# Written by Boris Bonev, Nov. 2020

## data structure
mutable struct HssMatrix{T<:Number} <: AbstractMatrix{T}
  # 2x2 recursive block structure for branchnodes
  A11       ::Union{HssMatrix{T}, Nothing}
  A22       ::Union{HssMatrix{T}, Nothing}
  B12       ::Union{Matrix{T}, Nothing}
  B21       ::Union{Matrix{T}, Nothing}

  # indicators whether we are at the root- or leafnode
  rootnode  ::Bool
  leafnode  ::Bool

  # dimensions of its children
  m1        ::Integer
  n1        ::Integer
  m2        ::Integer
  n2        ::Integer

  # translation operators if we are at a branchnode
  R1        ::Union{Matrix{T}, Nothing}
  R2        ::Union{Matrix{T}, Nothing}
  W1        ::Union{Matrix{T}, Nothing}
  W2        ::Union{Matrix{T}, Nothing}

  # at the leaf level we store the diagonal block and generators directly
  D         ::Union{Matrix{T}, Nothing}
  U         ::Union{Matrix{T}, Nothing}
  V         ::Union{Matrix{T}, Nothing}

  # inner constructors
  HssMatrix{T}() where T = new{T}() # this is the dirty constructor that leaves stuff #undef
  # rootnode constructor
  function HssMatrix{T}(A11::HssMatrix{T}, B12::Matrix{T}, B21::Matrix{T}, A22::HssMatrix{T}) where T
    m1, n1 = size(A11)
    m2, n2 = size(A22)
    if m1 != size(B12,1) throw(ArgumentError("A11 and B12 must have same number of rows")) end
    if n1 != size(B21,2) throw(ArgumentError("A11 and B21 must have same number of columns")) end
    if m2 != size(B21,1) throw(ArgumentError("A22 and B21 must have same number of rows")) end
    if n2 != size(B12,2) throw(ArgumentError("A22 and B12 must have same number of columns")) end
    new{T}(A11, B12, B21, A22, true, false, m1, n1, m2, n2, ntuple(x->nothing, 7)...)
  end
  # leafnode constructor
  function HssMatrix{T}(D::Matrix{T}, U::Matrix{T}, V::Matrix{T}) where T
    if size(D,1) != size(U,1) throw(ArgumentError("D and U must have same number of rows")) end
    if size(D,2) != size(V,1) throw(ArgumentError("D and V must have same number of columns")) end
    new{T}(ntuple(x->nothing, 4)..., true, false, m1, n1, m2, n2, ntuple(x->nothing, 4)..., D, U, V)
  end
end

# make element type extraction work
eltype(::Type{HssMatrix{T}}) where {T} = T

## copy operators
# maybe this should be called deepcopy?
function copy(hssA::HssMatrix{T}) where {T}
  hssB = HssMatrix{T}()

  hssB.A11 = copy(hssA.A11)
  hssB.A22 = copy(hssA.A22)
  hssB.B12 = copy(hssA.B12)
  hssB.B21 = copy(hssA.B21)

  hssB.rootnode = hssA.rootnode
  hssB.leafnode = hssA.leafnode

  hssB.m1 = hssA.m1
  hssB.n1 = hssA.n1
  hssB.m2 = hssA.m2
  hssB.n2 = hssA.n2

  hssB.R1 = copy(hssA.R1)
  hssB.R2 = copy(hssA.R2)
  hssB.W1 = copy(hssA.W1)
  hssB.W2 = copy(hssA.W2)

  hssB.U = copy(hssA.U)
  hssB.V = copy(hssA.V)
  hssB.D = copy(hssA.D)

  return(hssB)
end

## conversion
#convert(::HssMatrix{T}, hssA::HssMatrix) where {T} = HssMatrix()

## Typecasting routines
# function HssMatrix{T<:Number}(hssA::HssMatrix{S}) where S
#   isdefined() ? HssMatrix
# end

## Overriding some standard routines

# Base.size
Base.size(hssA::HssMatrix) = hssA.leafnode ? size(hssA.D) : (hssA.m1+hssA.m2, hssA.n1+hssA.n2)
function Base.size(hssA::HssMatrix, dim::Integer)
  if dim == 1
    return hssA.m1+hssA.m2
  elseif dim == 2
    return hssA.n1+hssA.n2
  elseif dim <= 0
    error("arraysize: dimension out of range")
  else
    return 1
  end
end

# construct full matrix from HSS
function Base.Matrix(hssA::HssMatrix{T}) where {T}
  n = size(hssA,2)
  return hssA * Union{Matrix{T}, Nothing}(I, n, n)
end

# alternatively we can form the full matrix in a more straight-forward fashion