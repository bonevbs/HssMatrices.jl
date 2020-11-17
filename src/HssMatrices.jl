module HssMatrices

  using LinearAlgebra
  using InvertedIndices, DataStructures

  export HssMatrix, bisection_cluster, hss_from_cluster, hss_compress_direct

  include("./prrqr.jl")
  include("./cluster_trees.jl")
  include("./hss_matrix.jl")
  include("./compression.jl")
end
