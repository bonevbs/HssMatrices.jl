using AbstractTrees

## definition of datastructures
# recursive format for HSS matrices
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

## Optional enhancements, not entirely sure this is needed
# These next two definitions allow inference of the item type in iteration.
# (They are not sufficient to solve all internal inference issues, however.)
Base.eltype(::Type{<:TreeIterator{HssMatrix{T}}}) where T = HssMatrix{T}
Base.IteratorEltype(::Type{<:TreeIterator{HssMatrix{T}}}) where T = Base.HasEltype()

function AbstractTrees.children(node::HssMatrix)
  if isdefined(node, :A11)
    if isdefined(node, :A22)
      return (node.A11, node.A22)
    end
    return (node.A11,)
  end
  isdefined(node, :A11) && return (node.A22,)
  return ()
end

AbstractTrees.printnode(io::IO, node::HssMatrix) = print(io, typeof(node))
