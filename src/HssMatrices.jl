__precompile__()
module HssMatrices

  using LinearAlgebra
  using AbstractTrees

  # using InvertedIndices, DataStructures
  import Base.*

  global tol = 1e-9
  global reltol = true
  global leafsize = 32

  export tol, reltol, leafsize
  export HssMatrix, bisection_cluster, hss_from_cluster, hss_compress_direct
  export hss_generators
  export *

  include("./prrqr.jl")
  include("./cluster_trees.jl")
  include("./hss_matrix.jl")
  include("./compression.jl")
  include("./generators.jl")
  include("./matmul.jl")
end
