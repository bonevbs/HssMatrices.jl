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
bisection_cluster(n::Int, opts::HssOptions=HssOptions(); args...) = bisection_cluster(1:n, opts; args...)
function bisection_cluster(range::UnitRange{Int}, opts::HssOptions=HssOptions(); args...)
  opts = copy(opts; args...)
  chkopts!(opts)
  length(range) ≤ 0 && throw(ArgumentError("Index range must be larger or equal to 0"))
  _bisection_cluster(range, opts.leafsize)
end
function bisection_cluster(n::Tuple{Int,Int}, opts::HssOptions=HssOptions(); args...)
  opts = copy(opts; args...)
  chkopts!(opts)
  1 ≤ n[1] ≤ n[2] || throw(ArgumentError("Indices must be ordered and bigger than 1"))
  ClusterTree(1:n[2], _bisection_cluster(1:n[1], opts.leafsize), _bisection_cluster(n[1]+1:n[2], opts.leafsize))
end
function _bisection_cluster(range::UnitRange{Int}, leafsize::Int)
  node = ClusterTree(range)
  if length(range) > leafsize
    n = convert(Int, ceil(size(range,1)/2))
    node.left = _bisection_cluster(range[1:n], leafsize)
    node.right = _bisection_cluster(range[n+1:end], leafsize)
  end
  return node
end