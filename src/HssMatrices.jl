__precompile__()
module HssMatrices

  using LinearAlgebra
  using SparseArrays # introduce custom constructors from sparse matrices
  using AbstractTrees

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

  include("./hss_matrix.jl")
  include("./prrqr.jl")
  include("./cluster_trees.jl")
  include("./compression.jl")
  include("./basicops.jl")
  include("./generators.jl")
  include("./matmul.jl")
end
