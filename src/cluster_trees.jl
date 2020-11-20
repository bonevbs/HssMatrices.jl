## most of this is taken from the AbstractTrees example
mutable struct BinaryNode{T}
  data::T
  parent::BinaryNode{T}
  left::BinaryNode{T}
  right::BinaryNode{T}

  # Root constructor
  BinaryNode{T}(data) where T = new{T}(data)
  # Child node constructor
  BinaryNode{T}(data, parent::BinaryNode{T}) where T = new{T}(data, parent)
end
BinaryNode(data) = BinaryNode{typeof(data)}(data)

function leftchild(data, parent::BinaryNode)
  !isdefined(parent, :left) || error("left child is already assigned")
  node = typeof(parent)(data, parent)
  parent.left = node
end
function rightchild(data, parent::BinaryNode)
  !isdefined(parent, :right) || error("right child is already assigned")
  node = typeof(parent)(data, parent)
  parent.right = node
end

function AbstractTrees.children(node::BinaryNode)
  if isdefined(node, :left)
    if isdefined(node, :right)
      return (node.left, node.right)
    end
    return (node.left,)
  end
  isdefined(node, :right) && return (node.right,)
  return ()
end

AbstractTrees.printnode(io::IO, node::BinaryNode) = print(io, node.data)

## Optional enhancements
# These next two definitions allow inference of the item type in iteration.
# (They are not sufficient to solve all internal inference issues, however.)
Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()

## Functionality for cluster trees
function bisection_cluster(range::UnitRange{T}, leafsize::T) where {T <: Integer}
  range[end] >= range[1] || error("last index is not bigger than the first index")
  node = BinaryNode{typeof(range)}(range)
  if size(range,1) > leafsize
    n = convert(T, ceil(size(range,1)/2))
    node.left = bisection_cluster(range[1:n], leafsize)
    node.right = bisection_cluster(range[n+1:end], leafsize)
  end
  return node
end

## Not happy with the entire situation here, might rewrite this part of the code
# build empty hssB = HssMatrix from given row and column cluster trees 
function hss_from_cluster(rcl::BinaryNode{UnitRange{S}}, ccl::BinaryNode{UnitRange{S}}) where {S <: Integer}
  # use the recursive routine
  hssA = HssMatrix{Float64}()
  hss_from_cluster_rec!(hssA, rcl, ccl)
  hssA.rootnode = true
  return hssA
end

function hss_from_cluster!(hssA::HssMatrix{T}, rcl::BinaryNode{UnitRange{S}}, ccl::BinaryNode{UnitRange{S}}) where {T, S <: Integer}
  hss_from_cluster_rec!(hssA, rcl, ccl)
  return hssA
end

# recursive definition
function hss_from_cluster_rec!(hssA::HssMatrix{T}, rcl::BinaryNode{UnitRange{S}}, ccl::BinaryNode{UnitRange{S}}) where {T <: Number, S <: Integer}
  if isdefined(rcl, :left) && isdefined(ccl, :left)
    hssA.A11 = HssMatrix{T}()
    hss_from_cluster_rec!(hssA.A11, rcl.left, ccl.left)
    hssA.A11.rootnode = false
    hssA.m1 = length(rcl.left.data)
    hssA.n1 = length(ccl.left.data)
  elseif isdefined(rcl, :left) || isdefined(ccl, :left)
    error("row and column cluster trees do not have matching structure")
  end
  if isdefined(rcl, :right) && isdefined(ccl, :right)
    hssA.A22 = HssMatrix{T}()
    hss_from_cluster_rec!(hssA.A22, rcl.right, ccl.right)
    hssA.A22.rootnode = false
    hssA.m2 = length(rcl.right.data)
    hssA.n2 = length(ccl.right.data)
  elseif isdefined(rcl, :right) || isdefined(ccl, :right)
    error("row and column cluster trees do not have matching structure")
  end
  if !isdefined(rcl, :left) && !isdefined(rcl, :right)
    hssA.leafnode = true
  end
end