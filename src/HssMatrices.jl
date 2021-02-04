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
  import Base.+, Base.-, Base.*, Base.Matrix, Base.copy, Base.size

  global tol = 1e-9
  global reltol = true
  global leafsize = 32

  #export tol, reltol, leafsize
  # hssmatrix.jl
  export HssLeaf, HssNode, HssMatrix, isleaf, isbranch, hssrank, full, prune_leaves!
  # prrqr.jl
  export prrqr!, truncate_block!
  # binarytree.jl and clustertree.jl
  export BinaryNode, leftchild, rightchild, isleaf, isbranch, bisection_cluster
  # compression.jl
  export compress_direct, recompress!
  # generators.jl  
  export generators, orthonormalize_generators!
  # matmul.jl
  # ulvfactor.jl
  export ulvfactsolve
  # hssdivide.jl
  export hssldivide!, _ulvfactor_leaves!
  # constructors.jl
  export lowrank2hss
  # visualization.jl
  export plotranks, pcolor

  include("hssmatrix.jl")
  include("prrqr.jl")
  include("binarytree.jl")
  include("clustertree.jl")
  include("compression.jl")
  include("generators.jl")
  include("matmul.jl")
  include("ulvfactor.jl")
  include("hssdivide.jl")
  include("constructors.jl")
  include("visualization.jl")
end
