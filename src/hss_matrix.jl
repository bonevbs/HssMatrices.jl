### Definitions of datastructures and basic constructors and operators
# Written by Boris Bonev, Nov. 2020

## data structure
mutable struct HssMatrix{T<:Number} <: AbstractMatrix{T}
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
end

# make element type extraction work
eltype(::Type{HssMatrix{T}}) where {T} = T

## copy operators
# maybe this should be called deepcopy?
function copy(hssA::HssMatrix{T}) where {T}
  hssB = HssMatrix{T}()

  if isdefined(hssA, :A11); hssB.A11 = copy(hssA.A11); end
  if isdefined(hssA, :A22); hssB.A22 = copy(hssA.A22); end
  if isdefined(hssA, :B12); hssB.B12 = copy(hssA.B12); end
  if isdefined(hssA, :B21); hssB.B21 = copy(hssA.B21); end

  if isdefined(hssA, :rootnode); hssB.rootnode = hssA.rootnode; end
  if isdefined(hssA, :leafnode); hssB.leafnode = hssA.leafnode; end

  if isdefined(hssA, :m1); hssB.m1 = hssA.m1; end
  if isdefined(hssA, :n1); hssB.n1 = hssA.n1; end
  if isdefined(hssA, :m2); hssB.m2 = hssA.m2; end
  if isdefined(hssA, :n2); hssB.n2 = hssA.n2; end

  if isdefined(hssA, :R1); hssB.R1 = copy(hssA.R1); end
  if isdefined(hssA, :R2); hssB.R2 = copy(hssA.R2); end
  if isdefined(hssA, :W1); hssB.W1 = copy(hssA.W1); end
  if isdefined(hssA, :W2); hssB.W2 = copy(hssA.W2); end

  if isdefined(hssA, :U); hssB.U = copy(hssA.U); end
  if isdefined(hssA, :V); hssB.V = copy(hssA.V); end
  if isdefined(hssA, :D); hssB.D = copy(hssA.D); end

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
  return hssA * Matrix{T}(I, n, n)
end

# alternatively we can form the full matrix in a more straight-forward fashion