### Definition of Binary trees
# This was mostly copied over from the AbstractTrees example, credits are due
#
# Copied from AbstractTrees.jl Jan. 2021

## most of this is taken from the AbstractTrees example
mutable struct BinaryNode{T}
  data::Union{T, Nothing}
  left::Union{BinaryNode{T}, Nothing}
  right::Union{BinaryNode{T}, Nothing}

  # Root constructor
  BinaryNode{T}() where T = new{T}(nothing, nothing, nothing)
  BinaryNode{T}(data) where T = new{T}(data, nothing, nothing)
  BinaryNode{T}(data, left::BinaryNode{T}, right::BinaryNode{T}) where T = new{T}(data, left, right)
  BinaryNode{T}(left::BinaryNode{T}, right::BinaryNode{T}) where T = new{T}(nothing, left, right)
end
BinaryNode(data) = BinaryNode{typeof(data)}(data)
BinaryNode(left::BinaryNode{T}, right::BinaryNode{T}) where T = BinaryNode{T}(left, right)
BinaryNode(data, left::BinaryNode, right::BinaryNode) = BinaryNode{typeof(data)}(data, left, right)

isleaf(node::BinaryNode) = isnothing(node.left) && isnothing(node.right)
isbranch(node::BinaryNode) = !isnothing(node.left) && !isnothing(node.right)

depth(node::BinaryNode) = _depth(node, 1)
function _depth(node::BinaryNode, level)
  if isleaf(node)
    return level
  elseif !isnothing(node.left)
    if !isnothing(node.right)
      return max(_depth(node.left, level+1), _depth(node.right, level+1))
    else
      return _depth(node.left, level+1)
    end
  else
    return _depth(node.right, level+1)
  end
end

function AbstractTrees.children(node::BinaryNode)
  if !isnothing(node.left)
    if !isnothing(node.right)
      return (node.left, node.right)
    end
    return (node.left,)
  end
  !isnothing(node.right) && return (node.right,)
  return ()
end

# routine to check equivalence of tree structures
function compatible(node1::BinaryNode, node2::BinaryNode)
  if isnothing(node1.left) ⊻ isnothing(node2.left) return false end
  if isnothing(node1.right) ⊻ isnothing(node2.right) return false end
  if !isnothing(node1.left) && !compatible(node1.left, node2.left) return false end
  if !isnothing(node1.right) && !compatible(node1.right, node2.right) return false end
  return true
end

==(node1::BinaryNode, node2::BinaryNode) = (node1.data == node2.data) && (node1.left == node2.left) && (node1.right == node2.right)
!=(node1::BinaryNode, node2::BinaryNode) = (node1.data != node2.data) || (node1.left != node2.left) || (node1.right != node2.right)

## Optional enhancements
# These next two definitions allow inference of the item type in iteration.
Base.eltype(::Type{BinaryNode{T}}) where T = T
Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()

# routines for displaying trees
AbstractTrees.printnode(io::IO, node::BinaryNode) = print(io, node.data)
Base.show(io::IO, node::BinaryNode) = print(io, "BinaryNode{$(eltype(node))}")

# Implement iteration over the immediate children of a node
function Base.iterate(node::BinaryNode)
  isdefined(node, :left) && return (node.left, false)
  isdefined(node, :right) && return (node.right, true)
  return nothing
end
function Base.iterate(node::BinaryNode, state::Bool)
  state && return nothing
  isdefined(node, :right) && return (node.right, true)
  return nothing
end
Base.IteratorSize(::Type{BinaryNode{T}}) where T = Base.SizeUnknown()
#Base.eltype(::Type{BinaryNode{T}}) where T = BinaryNode{T}

# ## Things we need to define to leverage the native iterator over children
# ## for the purposes of AbstractTrees.
# # Set the traits of this kind of tree
# Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
# Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()
# AbstractTrees.parentlinks(::Type{BinaryNode{T}}) where T = AbstractTrees.StoredParents()
# AbstractTrees.siblinglinks(::Type{BinaryNode{T}}) where T = AbstractTrees.StoredSiblings()
# # Use the native iteration for the children
# AbstractTrees.children(node::BinaryNode) = node

# Base.parent(root::BinaryNode, node::BinaryNode) = isdefined(node, :parent) ? node.parent : nothing

# function AbstractTrees.nextsibling(tree::BinaryNode, child::BinaryNode)
#   isdefined(child, :parent) || return nothing
#   p = child.parent
#   if isdefined(p, :right)
#       child === p.right && return nothing
#       return p.right
#   end
#   return nothing
# end