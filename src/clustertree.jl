### Definition of Cluster Trees
# This was mostly copied over from the AbstractTrees example, credits are due
#
# Copied from AbstractTrees.jl Jan. 2021

if !isdefined(@__MODULE__, :BinaryNode)
  include("binarytree.jl")
end

const ClusterTree = BinaryNode{UnitRange{Int}}

## Functionality for cluster trees
# Create a simple cluster tree based on 
function bisection_cluster(range::UnitRange{Int}, leafsize::Int)
  if length(range) <= 0 throw(ArgumentError("Index range must be longer than 0")) end
  node = ClusterTree(range)
  if length(range) > leafsize
    n = convert(Int, ceil(size(range,1)/2))
    node.left = bisection_cluster(range[1:n], leafsize)
    node.right = bisection_cluster(range[n+1:end], leafsize)
  end
  return node
end