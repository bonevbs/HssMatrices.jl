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
  using Plots # in the future, move to RecipesBase instead

  # load BLAS/LAPACK routines only used within the library# load efficient BLAS and LAPACK routines for factorizations
  import LinearAlgebra.LAPACK.geqlf!
  import LinearAlgebra.LAPACK.gelqf!
  import LinearAlgebra.LAPACK.ormql!
  import LinearAlgebra.LAPACK.ormlq!
  import LinearAlgebra.BLAS.ger!
  import LinearAlgebra.BLAS.trsm

  # using InvertedIndices, DataStructures
  import Base.+, Base.-, Base.*, Base.Matrix, Base.copy, Base.size
  import LinearAlgebra.ldiv!

  global tol = 1e-9
  global reltol = true
  global leafsize = 64

  #export tol, reltol, leafsize
  # hssmatrix.jl
  export HssLeaf, HssNode, HssMatrix, isleaf, isbranch, hssrank, full, checkdims, prune_leaves!
  # prrqr.jl
  export prrqr!
  # binarytree.jl
  export BinaryNode, leftchild, rightchild, isleaf, isbranch
  # clustertree.jl
  export bisection_cluster, cluster
  # compression.jl
  export compress_direct, recompress!
  # generators.jl  
  export generators, orthonormalize_generators!
  # matmul.jl
  # ulvfactor.jl
  export ulvfactsolve
  # hssdivide.jl
  export ldiv!, _ulvfactor_leaves!
  # constructors.jl
  export lowrank2hss
  # visualization.jl
  export plotranks, pcolor

  include("hssmatrix.jl")
  include("prrqr2.jl")
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
