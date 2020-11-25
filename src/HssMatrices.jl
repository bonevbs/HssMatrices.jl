### HssMatrices.jl module
# Written by Boris Bonev, Nov. 2020
__precompile__()
module HssMatrices

  using LinearAlgebra
  using SparseArrays # introduce custom constructors from sparse matrices
  using AbstractTrees
  #using RecipesBase # in the future, move to RecipesBase
  using Plots

  # using InvertedIndices, DataStructures
  import Base.*, Base.Matrix

  global tol = 1e-9
  global reltol = true
  global leafsize = 32

  export prrqr!, truncate_block!
  export tol, reltol, leafsize
  export HssMatrix, bisection_cluster, hss_from_cluster, hss_compress_direct, hss_recompress!
  export hssrank
  export generators, orthonormalize_generators!
  export *
  export plotranks

  include("hss_matrix.jl")
  include("prrqr.jl")
  include("cluster_trees.jl")
  include("compression.jl")
  include("basicops.jl")
  include("generators.jl")
  include("matmul.jl")
  include("visualization.jl")
end
