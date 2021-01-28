### Definition of Binary trees
# This was mostly copied over from the AbstractTrees example, credits are due
#
# Copied from AbstractTrees.jl Jan. 2021

## most of this is taken from the AbstractTrees example
mutable struct BinaryNode{T}
  data::T
  parent::BinaryNode{T}
  left::BinaryNode{T}
  right::BinaryNode{T}

  # Root constructor
  BinaryNode{T}(data) where T = new{T}(data)
  # Child constructor
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

## Optional enhancements
# These next two definitions allow inference of the item type in iteration.
Base.eltype(::Type{BinaryNode{T}}) where T = T
Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()

# routines for displaying trees
AbstractTrees.printnode(io::IO, node::BinaryNode) = print(io, node.data)
Base.show(io::IO, node::BinaryNode) = print(io, "BinaryNode{$(eltype(node))}")