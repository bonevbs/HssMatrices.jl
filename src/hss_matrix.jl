### Definitions of datastructures and basic constructors and operators
# Written by Boris Bonev, Nov. 2020

# recursive structure for HSS matrices
mutable struct HssMatrix{T<:Number}
  # 2x2 recursive block structure for branchnodes
  A11       ::HssMatrix{T}
  A22       ::HssMatrix{T}
  B12       ::Matrix{T}
  B21       ::Matrix{T}

  # indicators whether we are at the root- or leafnode
  rootnode  ::Bool
  leafnode  ::Bool

  # dimensions of its children
  m1        ::Integer
  n1        ::Integer
  m2        ::Integer
  n2        ::Integer

  # translation operators if we are at a branchnode
  R1        ::Matrix{T}
  R2        ::Matrix{T}
  W1        ::Matrix{T}
  W2        ::Matrix{T}

  # at the leaf level we store the diagonal block and generators directly
  U         ::Matrix{T}
  V         ::Matrix{T}
  D         ::Matrix{T}

  # inner constructor
  HssMatrix{T}() where T = new{T}()
  HssMatrix() = HssMatrix{Number}()
end

# make element type extraction work
eltype(::Type{<:HssMatrix{T}}) where {T} = T

## Typecasting routines
# function HssMatrix{T<:Number}(hssA::HssMatrix{S}) where S
#   isdefined() ? HssMatrix
# end

## Overriding some standard routines

# Base.size
Base.size(hssA::HssMatrix) = hssA.leafnode ? size(hssA.D) : hssA.m1+hssA.m2, hssA.n1+hssA.n2
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

# typecasting to full matrix
function Base.Matrix(hssA::HssMatrix{T}) where {T}
  n = size(hssA,2)
  return hssA * Matrix{T}(I, n, n)
end