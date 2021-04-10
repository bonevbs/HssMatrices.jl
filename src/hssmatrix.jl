### Definitions of datastructures and basic constructors and operators
# Written by Boris Bonev, Jan. 2021

## new datastructure which is the old datastructure
mutable struct HssMatrix{T<:Number} <: AbstractMatrix{T}
  # toggles for the type of node
  leafnode::Bool
  rootnode::Bool

  # fields for leaf nodes
  D ::Matrix{T}
  U ::Matrix{T}
  V ::Matrix{T}

  # fields for branch nodes
  A11 ::HssMatrix{T}
  A22 ::HssMatrix{T}
  B12 ::Matrix{T}
  B21 ::Matrix{T}

  sz1 ::Tuple{Int, Int}
  sz2 ::Tuple{Int, Int}

  R1 ::Matrix{T}
  W1 ::Matrix{T}
  R2 ::Matrix{T}
  W2 ::Matrix{T}

  # internal constructors for leaf nodes
  function HssMatrix(D::Matrix{T}, rootnode=true) where T
    m, n = size(D)
    new{T}(true, rootnode, D, Matrix{T}(undef, m, 0), Matrix{T}(undef, n, 0))
  end
  function HssMatrix(D::AbstractMatrix{T}, U::AbstractMatrix{T}, V::AbstractMatrix{T}, rootnode=false) where T
    if size(D,1) != size(U,1) throw(ArgumentError("D and U must have same number of rows")) end
    if size(D,2) != size(V,1) throw(ArgumentError("D and V must have same number of columns")) end
    new{T}(true, rootnode, D, U, V)
  end
  # internal constructors for branch nodes
  function HssMatrix(A11::HssMatrix{T}, A22::HssMatrix{T}, B12::AbstractMatrix{T}, B21::AbstractMatrix{T}, rootnode=true) where T
    kr1, kw1 = gensize(A11); kr2, kw2 = gensize(A22)
    hssA = new{T}(false, rootnode)
    hssA.A11 = A11; hssA.A22 = A22
    hssA.B12 = B12; hssA.B21 = B21
    hssA.sz1 = size(A11); hssA.sz2 = size(A22)
    hssA.R1 = Matrix{T}(undef,kr1,0); hssA.W1 = Matrix{T}(undef,kw1,0)
    hssA.R2 = Matrix{T}(undef,kr2,0); hssA.W2 = Matrix{T}(undef,kw2,0)
    return hssA
  end
  function HssMatrix(A11::HssMatrix{T}, A22::HssMatrix{T}, B12::AbstractMatrix{T}, B21::AbstractMatrix{T}, 
    R1::AbstractMatrix{T}, W1::AbstractMatrix{T}, R2::AbstractMatrix{T}, W2::AbstractMatrix{T}, rootnode=false) where T
    if size(R1,2) != size(R2,2) throw(DimensionMismatch("R1 and R2 must have same number of columns")) end
    if size(W1,2) != size(W2,2) throw(DimensionMismatch("W1 and W2 must have same number of rows")) end
    hssA = new{T}(false, rootnode)
    hssA.A11 = A11; hssA.A22 = A22
    hssA.B12 = B12; hssA.B21 = B21
    hssA.sz1 = size(A11); hssA.sz2 = size(A22)
    hssA.R1 = R1; hssA.W1 = W1
    hssA.R2 = R2; hssA.W2 = W2
    return hssA
  end
end

# custom constructors which are calling the compression algorithms
function hss(A::AbstractMatrix, opts=HssOptions(Float64); args...)
  opts = copy(opts; args...)
  chkopts!(opts)
  return hss(A, bisection_cluster(size(A,1), leafsize=opts.leafsize), bisection_cluster(size(A,2), leafsize=opts.leafsize); args...)
end
hss(A::Matrix, rcl::ClusterTree, ccl::ClusterTree; args...) = compress(A, rcl, ccl; args...)
function hss(A::AbstractSparseMatrix, rcl::ClusterTree, ccl::ClusterTree; args...)
  m, n = size(A)
  # estimate rank by assuming that the non-zero entries are clustered on the diagonal
  nl = nleaves(rcl)
  m0 = Int(ceil(m/nl))
  n0 = Int(ceil(n/nl))
  kest = max(nnz(A) - nl*m0*n0,0)
  return randcompress_adaptive(A, rcl, ccl; kest=kest, args...)
end
hss(A::LinearMap, rcl::ClusterTree, ccl::ClusterTree; args...) = randcompress_adaptive(A, rcl, ccl; args...)

@inline isleaf(hssA::HssMatrix) = hssA.leafnode # check whether making this inline speeds up things ?
@inline isbranch(hssA::HssMatrix) = !hssA.leafnode
@inline isroot(hssA::HssMatrix) = hssA.rootnode

@inline ishss(A::AbstractMatrix) = typeof(A) <: HssMatrix

size(hssA::HssMatrix) = isleaf(hssA) ? size(hssA.D) : hssA.sz1 .+ hssA.sz2
size(hssA::HssMatrix, dim::Integer) = size(hssA)[dim]

show(io::IO, hssA::HssMatrix) = print(io, "$(size(hssA,1))x$(size(hssA,2)) HssMatrix{$(eltype(hssA))}")

# implement sorted access to entries via recursion
# in the long run we might want to return an HssMatrix when we access via getindex
# TODO: add @boundscheck for the bound checking
getindex(hssA::HssMatrix, i::Int, j::Int) = _getidx(hssA, i, j)[1]
getindex(hssA::HssMatrix, i::Int, j::AbstractRange) = getindex(hssA, [i], j)[:]
getindex(hssA::HssMatrix, i::AbstractRange, j::Int) = getindex(hssA, i, [j])[:]
getindex(hssA::HssMatrix, i::AbstractRange, j::AbstractRange) = getindex(hssA, collect(i), collect(j))
getindex(hssA::HssMatrix, ::Colon, ::Colon) = full(hssA)
getindex(hssA::HssMatrix, i, ::Colon) = getindex(hssA, i, 1:size(hssA,2))
getindex(hssA::HssMatrix, ::Colon, j) = getindex(hssA, 1:size(hssA,1), j)
function getindex(hssA::HssMatrix{T}, i::Vector{Int}, j::Vector{Int}) where T
  m, n  = size(hssA)
  ip = sortperm(i); jp = sortperm(j)
  if (length(i) == 0 || length(j) == 0) return Matrix{T}(undef, length(i), length(j)) end
  return full(_getidx(hssA, i[ip], j[jp]))[invperm(ip), invperm(jp)]
end

# First construct a sub-HSS matrix and then call full()
function _getidx(hssA::HssMatrix, i::Vector{Int}, j::Vector{Int})
  if isleaf(hssA)
    HssMatrix(hssA.D[i,j], hssA.U[i,:], hssA.V[j,:])
  else
    m1, n1 = hssA.sz1
    i1 = i[i .<= m1]; j1 = j[j .<= n1]
    i2 = i[i .> m1] .- m1; j2 = j[j .> n1] .- n1
    A11 = _getidx(hssA.A11, i1, j1)
    A22 = _getidx(hssA.A22, i2, j2)
    return HssMatrix(A11, A22, hssA.B12, hssA.B21, hssA.R1, hssA.W1, hssA.R2, hssA.W2)
  end
end
# for individual indices, it should be faster to access them in the following way
function _getidx(hssA::HssMatrix, i::Int, j::Int)
  if isleaf(hssA)
    return hssA.D[i,j]
  else
    m1, n1 = hssA.sz1
    if i <= m1
      if j <= n1
        return _getidx(hssA.A11, i, j)
      else
        U1 = _getindex_colgenerator(hssA.A11, i)
        V2 = _getindex_rowgenerator(hssA.A22, j-n1)
        return dot(U1*hssA.B12, V2)
      end
    else
      if j <= n1
        U2 = _getindex_colgenerator(hssA.A22, i-m1)
        V1 = _getindex_rowgenerator(hssA.A11, j)
        return dot(U2*hssA.B21, V1)
      else
        return _getidx(hssA.A22, i-m1, j-n1)
      end
    end
  end
end

# perhaps, this should be deepcopy?
function copy(hssA::HssMatrix)
  if isleaf(hssA)
    return HssMatrix(copy(hssA.D), copy(hssA.U), copy(hssA.V), hssA.rootnode)
  else
    return HssMatrix(copy(hssA.A11), copy(hssA.A22), copy(hssA.B12), copy(hssA.B21), copy(hssA.R1), copy(hssA.W1), copy(hssA.R2), copy(hssA.W2), hssA.rootnode)
  end
end

# introduce lazy adjoint, then all this copying won't be necessary
function adjoint(hssA::HssMatrix)
  if isleaf(hssA)
    return HssMatrix(copy(adjoint(hssA.D)), copy(hssA.V), copy(hssA.U), hssA.rootnode)
  else
    return HssMatrix(adjoint(hssA.A11), adjoint(hssA.A22), copy(adjoint(hssA.B21)), copy(adjoint(hssA.B12)), copy(hssA.W1), copy(hssA.R1), copy(hssA.W2), copy(hssA.R2), hssA.rootnode)
  end
end

# transpose
function transpose(hssA::HssMatrix)
  if isleaf(hssA)
    return HssMatrix(copy(transpose(hssA.D)), copy(hssA.V), copy(hssA.U), hssA.rootnode)
  else
    return HssMatrix(transpose(hssA.A11), transpose(hssA.A22), copy(transpose(hssA.B21)), copy(transpose(hssA.B12)), copy(hssA.W1), copy(hssA.R1), copy(hssA.W2), copy(hssA.R2), hssA.rootnode)
  end
end

# Define Matlab-like convenience functions, which are used throughout the library
#blkdiagm(A::Matrix, B::Matrix) = [A zeros(size(A,1), size(B,2)); zeros(size(B,1), size(A,2)) B]
#blkdiagm(A::Matrix... ) = blkdiagm(A[1], blkdiagm(A[2:end]...))
blkdiagm(A::Matrix...) = cat(A[1:end]..., dims=(1,2))

## basic algebraic operations (taken and modified from LowRankApprox.jl)
for op in (:+,:-)
  @eval begin
    $op(hssA::HssMatrix) = isleaf(hssA) ? HssMatrix($op(hssA.D), hssA.U, hssA.V) : HssMatrix($op(hssA.A11), $op(hssA.A22), $op(hssA.B12), $op(hssA.B21), hssA.R1, hssA.W1, hssA.R2, hssA.W2)

    $op(a::Bool, hssA::HssMatrix{Bool}) = error("Not callable")
    $op(L::HssMatrix{Bool}, a::Bool) = error("Not callable")

    function $op(hssA::HssMatrix, hssB::HssMatrix)
      size(hssA) == size(hssB) || throw(DimensionMismatch("A has dimensions $(size(hssA)) but B has dimensions $(size(hssB))"))
      if isleaf(hssA) && isleaf(hssB)
        return HssMatrix($op(hssA.D, hssB.D), [hssA.U hssB.U], [hssA.V hssB.V], isroot(hssA) && isroot(hssB))
      elseif isbranch(hssA) && isbranch(hssB)
        hssA.sz1 == hssB.sz1 || throw(DimensionMismatch("A11 has dimensions $(hssA.sz1) but B11 has dimensions $(hssB.sz1)"))
        hssA.sz2 == hssB.sz2 || throw(DimensionMismatch("A22 has dimensions $(hssA.sz2) but B22 has dimensions $(hssA.sz2)"))
        return HssMatrix($op(hssA.A11, hssB.A11), $op(hssA.A22, hssB.A22), blkdiagm(hssA.B12, $op(hssB.B12)), blkdiagm(hssA.B21, $op(hssB.B21)),blkdiagm(hssA.R1, hssB.R1), blkdiagm(hssA.W1, hssB.W1), blkdiagm(hssA.R2, hssB.R2), blkdiagm(hssA.W2, hssB.W2), isroot(hssA) && isroot(hssB))
      else
        throw(ArgumentError("Cannot add leaves and branches."))
      end
    end
  end
end

# matrix division involving HSS matrices
\(hssA::HssMatrix, B::Matrix) = ulvfactsolve(hssA, B)
\(hssA::HssMatrix, hssB::HssMatrix) = ldiv!(hssA, copy(hssB))
/(A::Matrix, hssB::HssMatrix) = ulvfactsolve(hssB', collect(A'))'
/(hssA::HssMatrix, hssB::HssMatrix) = rdiv!(copy(hssA), hssB)

# Scalar multiplication
*(hssA::HssMatrix, a::Number) = *(a, hssA)
function *(a::Number, hssA::HssMatrix)
  if isleaf(hssA)
    return HssMatrix(a*hssA.D, hssA.U, hssA.V, isroot(hssA))
  else
    return HssMatrix(a*hssA.A11, a*hssA.A22, a*hssA.B12, a*hssA.B21, hssA.R1, hssA.W1, hssA.R2, hssA.W2, isroot(hssA))
  end
end

## Some more fundamental operations
# compute the HSS rank
hssrank(hssA::HssMatrix) = isleaf(hssA) ? 0 : max(hssrank(hssA.A11), hssrank(hssA.A22), size(hssA.B12)..., size(hssA.B21)...)

# compute the sizeof the underlying generators
function gensize(hssA::HssMatrix)
  if isleaf(hssA)
    return size(hssA.U,2), size(hssA.V,2)
  else
    (kr = size(hssA.R1,2)) == size(hssA.R2,2) || throw(DimensionMismatch("dimensions of column-translators do not match"))
    (kw = size(hssA.W1,2)) == size(hssA.W2,2) || throw(DimensionMismatch("dimensions of row-translators do not match"))
    return kr, kw
  end
end

# function that returns alternative HssNode acting as rootnode
# this makes multiplication etc. safe if we use subblocks of HSS matrices
rooted(hssA::HssMatrix) = isleaf(hssA) ? HssMatrix(hssA.D, true) : HssMatrix(hssA.A11, hssA.A22, hssA.B12, hssA.B21, true)

# return a full matrix (hopefully efficient implementation with pre-allocated memory)
Matrix(hssA::HssMatrix) = full(hssA)
function full(hssA::HssMatrix{T}) where T
  if isleaf(hssA)
    return hssA.D
  else
    m, n = size(hssA)
    k = hssrank(hssA)
    A = Matrix{T}(undef, m, n)
    U = Matrix{T}(undef, m, k)
    V = Matrix{T}(undef, n, k)
    _full!(hssA, A, U, V, 0, 0; rootnode=true)
  end
  return A
end
# function that expands hss matrix using pre-allocated memory
function _full!(hssA::HssMatrix, A::Matrix,  U::Matrix, V::Matrix, ro::Int, co::Int; rootnode=false)
  if isleaf(hssA)
    m, n = size(hssA)
    A[ro+1:ro+m, co+1:co+n] = hssA.D
    U[ro+1:ro+m, 1:size(hssA.U,2)] = hssA.U
    V[co+1:co+n, 1:size(hssA.V,2)] = hssA.V
  else
    m1, n1 = hssA.sz1; m2, n2 = hssA.sz2
    ru1, rv1 = gensize(hssA.A11)
    ru2, rv2 = gensize(hssA.A22)
    _full!(hssA.A11, A, U, V, ro, co)
    _full!(hssA.A22, A, U, V, ro+m1, co+n1)
    A[ro+1:ro+m1, co+n1+1:co+n1+n2] = U[ro+1:ro+m1, 1:ru1]*hssA.B12*V[co+n1+1:co+n1+n2, 1:rv2]'
    A[ro+m1+1:ro+m1+m2, co+1:co+n1] = U[ro+m1+1:ro+m1+m2, 1:ru2]*hssA.B21*V[co+1:co+n1, 1:rv1]'
    if !rootnode
      U[ro+1:ro+m1, 1:size(hssA.R1,2)] = U[ro+1:ro+m1, 1:ru1]*hssA.R1
      U[ro+m1+1:ro+m1+m2, 1:size(hssA.R2,2)] = U[ro+m1+1:ro+m1+m2, 1:ru2]*hssA.R2
      V[co+1:co+n1, 1:size(hssA.W1,2)] = V[co+1:co+n1, 1:rv1]*hssA.W1
      V[co+n1+1:co+n1+n2, 1:size(hssA.W2,2)] = V[co+n1+1:co+n1+n2, 1:rv2]*hssA.W2
    end
  end
end

# useful routine to check whether dimensions are compatible
checkdims(hssA::HssMatrix)= _checkdims(hssA, 1)[1]
function _checkdims(hssA::HssMatrix, i::Int)
  if isleaf(hssA)
    compatible = (size(hssA.D,1) == size(hssA.U,1)) && (size(hssA.D,2) == size(hssA.V,1))
    if !compatible println("dimensions don't match in node ", i) end
    return compatible, i+1
  else
    comp1, i = _checkdims(hssA.A11, i)
    comp2, i = _checkdims(hssA.A22, i)
    r1, w1 = gensize(hssA.A11); r2, w2 = gensize(hssA.A22)
    compatible = (r1 == size(hssA.R1,1)) && (r2 == size(hssA.R2,1)) && (w1 == size(hssA.W1,1)) && (w2 == size(hssA.W2,1))
    if !compatible println("dimensions don't match in node ", i) end
    return (compatible && comp1 && comp2), i+1
  end
end

# remove leaves on the bottom level
function prune_leaves!(hssA::HssMatrix)
  if isleaf(hssA.A11) && isleaf(hssA.A22)
    return HssMatrix(_hssleaf(hssA)..., hssA.rootnode)
  else
    hssA.A11 = prune_leaves!(hssA.A11)
    hssA.A22 = prune_leaves!(hssA.A22)
    return hssA
  end
end

# returns D, U and V. replaces _full function
function _hssleaf(hssA::HssMatrix)
  if isleaf(hssA)
    return hssA.D, hssA.U, hssA.V
  else
    A11, U1, V1 = _hssleaf(hssA.A11)
    A22, U2, V2 = _hssleaf(hssA.A22)
    return [A11 U1*hssA.B12*V2'; U2*hssA.B21*V1' A22], [U1*hssA.R1; U2*hssA.R2], [V1*hssA.W1; V2*hssA.W2]
  end
end

## write function that extracts the clustwer tree from an HSS matrix
cluster(hssA) = _cluster(hssA, 0, 0)
function _cluster(hssA::HssMatrix, ro::Int, co::Int)
  if isleaf(hssA)
    m, n = size(hssA)
    return ClusterTree(ro .+ (1:m)), ClusterTree(co .+ (1:n))
  else
    rcl1, ccl1 = _cluster(hssA.A11, ro, co)
    rcl2, ccl2 = _cluster(hssA.A22, rcl1.data[end], ccl1.data[end])
    return ClusterTree(ro+1:rcl2.data[end], rcl1, rcl2), ClusterTree(co+1:ccl2.data[end], ccl1, ccl2)
  end
end