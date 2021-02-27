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
bisection_cluster(n::Int, opts::HssOptions=HssOptions(); args... ) = bisection_cluster(1:n, opts)
function bisection_cluster(range::UnitRange{Int}, opts::HssOptions=HssOptions(); args...)
  opts = copy(opts; args...)
  chkopts!(opts)
  length(range) ≤ 0 && throw(ArgumentError("Index range must be larger or equal to 0"))
  _bisection_cluster(range, opts.leafsize)
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

## write function that extracts the clustwer tree from an HSS matrix
cluster(hssA) = _cluster(hssA, 0, 0)
function _cluster(hssA::HssLeaf, co::Int, ro::Int)
  m, n = size(hssA)
  return ClusterTree(co.+(1:m)), ClusterTree(ro.+(1:n))
end
function _cluster(hssA::HssNode, co::Int, ro::Int)
  ccl1, rcl1 = _cluster(hssA.A11, co, ro)
  ccl2, rcl2 = _cluster(hssA.A22, ccl1.data[end], rcl1.data[end])
  return ClusterTree(co:ccl2.data[end], ccl1, ccl2), ClusterTree(ro:rcl2.data[end], rcl1, rcl2)
end

## TODO: write function to check cluster equality