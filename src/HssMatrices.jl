### HssMatrices.jl module
# A simple Julia package that allows working with HSS matrices
# The package aims to be simple, intuitive and efficient
# Written by Boris Bonev, Nov. 2020
__precompile__()
module HssMatrices

  # dependencies, trying to keep this list to a minimum if possible
  using LinearAlgebra
  using SparseArrays # introduce custom constructors from sparse matrices
  using AbstractTrees
  using DataStructures
  #using RecipesBase
  using Plots # in the future, move to RecipesBase

  # using InvertedIndices, DataStructures
  import Base.*, Base.+, Base.Matrix, Base.copy, Base.size

  global tol = 1e-9
  global reltol = true
  global leafsize = 32

  export HssLeaf, HssNode

  #export prrqr!, truncate_block!
  #export tol, reltol, leafsize
  #export HssMatrix, bisection_cluster, hss_from_cluster!, hss_compress_direct, hss_recompress!
  #export hssrank
  #export generators, orthonormalize_generators!
  #export *
  #export ulvfactor, ulvsolve, ulvfactsolve, ULVFactor
  #export plotranks, pcolor

  include("hss_matrix.jl")
  include("prrqr.jl")
  # include("cluster_trees.jl")
  # include("compression.jl")
  # include("basicops.jl")
  # include("generators.jl")
  # include("matmul.jl")
  # include("ulvfactor.jl")
  # include("visualization.jl")
end
