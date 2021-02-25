### This File contains all compression routines for the HssMatrices.jl package
#
# Direct compression
# as described in
# Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices.
# Numerical Linear Algebra with Applications, 17(6), 953–976. https://doi.org/10.1002/nla.691
#
# Recompression
# as described in
# Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices.
# Numerical Linear Algebra with Applications, 17(6), 953–976. https://doi.org/10.1002/nla.691
#
# Randomized compression
# as described in
# Martinsson, P. G. (2011). A Fast Randomized Algorithm for Computing a Hierarchically Semiseparable Representation of a Matrix.
# SIAM Journal on Matrix Analysis and Applications, 32(4), 1251–1274. https://doi.org/10.1137/100786617
#
# Re-Written by Boris Bonev, Jan. 2021

## Direct compression algorithm
# wrapper function that will be exported
function compress(A::Matrix{T}, rcl::ClusterTree, ccl::ClusterTree, opts::HssOptions=HssOptions(T); args...) where T
  opts = copy(opts; args...)
  chkopts!(opts)
  compatible(rcl, ccl) || throw(ArgumentError("row and column clusters are not compatible"))
  m = length(rcl.data); n = length(ccl.data)
  if size(A) != (m,n) throw(ArgumentError("size of row- and column-cluster-trees must match")) end
  Brow = Array{T}(undef, m, 0)
  Bcol = Array{T}(undef, 0, n)
  if isleaf(rcl) && isleaf(ccl)
    hssA, _, _ = _compress!(A, Brow, Bcol, rcl.data, ccl.data, opts.atol, opts.rtol)
  elseif isbranch(rcl) && isbranch(ccl)
    hssA, _, _ = _compress!(A, Brow, Bcol, rcl, ccl, opts.atol, opts.rtol)
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end
  return hssA
end

# leaf node function for compression
function _compress!(A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, rows::UnitRange{Int}, cols::UnitRange{Int}, atol::Float64, rtol::Float64) where T
  U, Brow = _compress_block!(Brow, atol, rtol)
  V, Bcol = _compress_block!(Bcol', atol, rtol) #TODO: write code that is better at dealing with Julia's lazy transpose
  return HssLeaf(A[rows, cols], U, V), Brow, copy(Bcol')
end

# branch node function for compression
function _compress!(A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, rcl::ClusterTree, ccl::ClusterTree, atol::Float64, rtol::Float64) where T
  m1 = length(rcl.left.data); m2 = length(rcl.right.data)
  n1 = length(ccl.left.data); n2 = length(ccl.right.data)

  rows1 = rcl.left.data; rows2 = rcl.right.data
  cols1 = ccl.left.data; cols2 = ccl.right.data

  # left node
  Brow1 = [A[rows1, cols2]  Brow[1:m1, :]]
  Bcol1 = [A[rows2, cols1]; Bcol[:, 1:n1]]
  if isleaf(rcl.left) && isleaf(ccl.left)
    A11, Brow1, Bcol1 = _compress!(A, Brow1, Bcol1, rows1, cols1, atol, rtol)
  elseif isbranch(rcl.left) && isbranch(ccl.left)
    A11, Brow1, Bcol1 = _compress!(A, Brow1, Bcol1, rcl.left, ccl.left, atol, rtol)
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end

  # right node
  Brow2 = [Bcol1[1:m2, :]  Brow[m1+1:end, :]]
  Bcol2 = [Brow1[:, 1:n2]; Bcol[:, n1+1:end]]
  if isleaf(rcl.right) && isleaf(ccl.right)
    A22, Brow2, Bcol2 = _compress!(A, Brow2, Bcol2, rows2, cols2, atol, rtol)
  elseif isbranch(rcl.right) && isbranch(ccl.right)
    A22, Brow2, Bcol2 = _compress!(A, Brow2, Bcol2, rcl.right, ccl.right, atol, rtol)
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end

  # figure out compressed off-diagonal blocks
  rm1 = size(Brow1,1); rn1 = size(Bcol1, 2)
  rm2 = size(Brow2,1); rn2 = size(Bcol2, 2)
  B12 = Bcol2[1:rm1, :]
  B21 = Brow2[:, 1:rn1]

  # clean up stuff from the front and form the composed HSS block row/col for compression
  Brow = [Brow1[:, n2+1:end]; Brow2[:, rn1+1:end]]
  Bcol = [Bcol1[m2+1:end, :]  Bcol2[rm1+1:end, :]]
  
  # do the actual compression and disentangle blocks of the translation operators
  R, Brow = _compress_block!(Brow, atol, rtol)
  R1 = R[1:rm1, :]
  R2 = R[rm1+1:end, :]
  
  X = copy(Bcol')
  W, Bcol = _compress_block!(copy(Bcol'), atol, rtol);
  #println(size(X))
  #println(norm(X - W*Bcol)/norm(X))
  Bcol = copy(Bcol')
  W1 = W[1:rn1, :]
  W2 = W[rn1+1:end, :]

  # call recursively the 
  hssA = HssNode(A11, A22, B12, B21, R1, W1, R2, W2)
  return hssA, Brow, Bcol
end

## Recompression algorithm
function recompress!(hssA::HssMatrix{T}, opts::HssOptions=HssOptions(T); args...) where T
  opts = copy(opts; args...)
  chkopts!(opts)
  rtol = opts.rtol; atol = opts.atol;

  if isleaf(hssA); return hssA; end

  # a prerequisite for this algorithm to work is that generators are orthonormal
  orthonormalize_generators!(hssA)

  # compress B12, B21 via something that resembles the SVD
  P1, S2 = _compress_block!(hssA.B12, atol, rtol)
  Q2, T1 = _compress_block!(copy(hssA.B12'), atol, rtol)
  P2, S1 = _compress_block!(hssA.B21, atol, rtol)
  Q1, T2 = _compress_block!(copy(hssA.B21'), atol, rtol)

  hssA.B12 = S2*Q2
  hssA.B21 = S1*Q1

  # fix dimensions of ghost-translators in rootnode
  hssA.R1 = hssA.R1[1:size(hssA.B12, 1),:]
  hssA.W1 = hssA.W1[1:size(hssA.B21, 2),:]
  hssA.R2 = hssA.R2[1:size(hssA.B21, 1),:]
  hssA.W2 = hssA.W2[1:size(hssA.B12, 2),:]

  # pass information to children and proceed recursively
  if isbranch(hssA.A11)
    hssA.A11.R1 = hssA.A11.R1*P1
    hssA.A11.R2 = hssA.A11.R2*P1
    hssA.A11.W1 = hssA.A11.W1*Q1
    hssA.A11.W2 = hssA.A11.W2*Q1
  elseif isleaf(hssA.A11)
    hssA.A11.U = hssA.A11.U*P1
    hssA.A11.V = hssA.A11.V*Q1
  end
  # repeat for A22
  if isbranch(hssA.A22)
    hssA.A22.R1 = hssA.A22.R1*P2
    hssA.A22.R2 = hssA.A22.R2*P2
    hssA.A22.W1 = hssA.A22.W1*Q2
    hssA.A22.W2 = hssA.A22.W2*Q2
  elseif isleaf(hssA.A22)
    hssA.A22.U = hssA.A22.U*P2
    hssA.A22.V = hssA.A22.V*Q2
  end

  # call recompression recursively
  if isbranch(hssA.A11)
    _recompress!(hssA.A11, hssA.B12, copy(hssA.B21'), atol, rtol)
  end
  if isbranch(hssA.A22)
    _recompress!(hssA.A22, hssA.B21, copy(hssA.B12'), atol, rtol)
  end

  return hssA
end

function _recompress!(hssA::HssNode{T}, Brow::Matrix{T}, Bcol::Matrix{T}, atol, rtol) where T
  # compress B12
  Brow1 = [hssA.B12  hssA.R1*Brow]
  Bcol2 = [hssA.B12' hssA.W2*Bcol]
  P1, S2 = _compress_block!(Brow1, atol, rtol)
  Q2, T1 = _compress_block!(Bcol2, atol, rtol)
  # get the original number of columns in B12
  rm1, rn2 = size(hssA.B12)
  hssA.B12 = S2[:,1:rn2]*Q2
  hssA.R1 = P1'*hssA.R1
  hssA.W2 = Q2'*hssA.W2
  # compress B21
  Brow2 = [hssA.B21  hssA.R2*Brow]
  Bcol1 = [hssA.B21' hssA.W1*Bcol]
  P2, S1 = _compress_block!(Brow2, atol, rtol)
  Q1, T2 = _compress_block!(Bcol1, atol, rtol)
  # get the original number of columns in B21
  rm2, rn1 = size(hssA.B21)
  hssA.B21 = S1[:,1:rn1]*Q1
  hssA.R2 = P2'*hssA.R2
  hssA.W1 = Q1'*hssA.W1
  # update generators of A11
  if isbranch(hssA.A11)
    hssA.A11.R1 = hssA.A11.R1*P1
    hssA.A11.R2 = hssA.A11.R2*P1
    hssA.A11.W1 = hssA.A11.W1*Q1
    hssA.A11.W2 = hssA.A11.W2*Q1
  elseif isleaf(hssA.A11)
    hssA.A11.U = hssA.A11.U*P1
    hssA.A11.V = hssA.A11.V*Q1
  end
  # update generators in A22
  if isbranch(hssA.A22)
    hssA.A22.R1 = hssA.A22.R1*P2
    hssA.A22.R2 = hssA.A22.R2*P2
    hssA.A22.W1 = hssA.A22.W1*Q2
    hssA.A22.W2 = hssA.A22.W2*Q2
  else isleaf(hssA.A22)
    hssA.A22.U = hssA.A22.U*P2
    hssA.A22.V = hssA.A22.V*Q2
  end

  # call recompression recursively
  if isbranch(hssA.A11)
    Brow1 = hcat(hssA.B12, S2[:,rn2+1:end])
    Bcol1 = hcat(hssA.B21', T2[:,rm2+1:end])
    _recompress!(hssA.A11, Brow1, Bcol1, atol, rtol)
  end
  if isbranch(hssA.A22)
    Brow2 = hcat(hssA.B21, S1[:,rn1+1:end])
    Bcol2 = hcat(hssA.B12', T1[:,rm1+1:end])
    _recompress!(hssA.A22, Brow2, Bcol2, atol, rtol)
  end

  return hssA
end

## Randomized compression algorithm
function randcompress(A::AbstractMatOrLinOp{T}, rcl::ClusterTree, ccl::ClusterTree, kest::Int, opts::HssOptions=HssOptions(T), args...) where T
  opts = copy(opts; args...)
  chkopts!(opts)
  m, n = size(A)
  compatible(rcl, ccl) || throw(ArgumentError("row and column clusters are not compatible"))
  if typeof(A) <: AbstractMatrix A = LinOp(A) end
  hssA = _extract_diagonal(A, rcl, ccl)

  # compute initial sampling
  k = kest; r = opts.noversampling;
  Ωcol = randn(n, k+r)
  Ωrow = randn(m, k+r)
  Scol = A*Ωcol # this should invoke the magic of the linearoperator.jl type
  Srow = A*Ωrow
  hssA = _randcompress!(hssA, A, Scol, Srow, Ωcol, Ωrow, 0, 0, opts.atol, opts.rtol; rootnode=true)
  return hssA
end

function randcompress_adaptive(A::AbstractMatOrLinOp{T}, rcl::ClusterTree, ccl::ClusterTree, opts::HssOptions=HssOptions(T); kest=10, args...) where T
  opts = copy(opts; args...)
  chkopts!(opts)
  m, n = size(A)
  maxrank = n
  compatible(rcl, ccl) || throw(ArgumentError("row and column clusters are not compatible"))
  if typeof(A) <: AbstractMatrix A = LinOp(A) end
  hssA = _extract_diagonal(A, rcl, ccl)

  # compute initial sampling
  k = kest; r = opts.noversampling; bs = opts.stepsize
  #bs = Integer(ceil(n*0.01)) # this should probably be better estimated
  Ωcol = randn(n, k+r)
  Ωrow = randn(m, k+r)
  Scol = A*Ωcol # this should invoke the magic of the linearoperator.jl type
  Srow = A*Ωrow
  failed = true

  while failed && k < maxrank
    # TODO: In further versions we might want to re-use the information gained during previous attempts
    hssA = _randcompress!(hssA, A, Scol, Srow, Ωcol, Ωrow, 0, 0, opts.atol, opts.rtol; rootnode=true)

    Ωcol_test = randn(n, bs)
    Ωrow_test = randn(m, bs)
    Scol_test = A*Ωcol_test
    Srow_test = A'*Ωrow_test

    #@infiltrate
    nrm = sqrt(1/bs)*norm(Scol_test)
    nrm_est = sqrt(1/bs)*norm(Scol_test - hssA*Ωcol_test)
    failed = nrm_est > opts.atol && nrm_est > opts.rtol*nrm

    if failed
      opts.verbose || println("Enlarging sampling space to ", k+bs)
      Ωcol = [Ωcol Ωcol_test]
      Ωrow = [Ωrow Ωrow_test]
      Scol = [Scol Scol_test]
      Srow = [Srow Srow_test]
      k = k+bs
    end
  end

  return hssA
end

# this function compresses given the sampling matrix of rank k
function _extract_diagonal(A::AbstractMatOrLinOp{T}, rcl::ClusterTree, ccl::ClusterTree) where T
  if isleaf(rcl) # only check row cluster as we have already checked cluster equality
    D = A[rcl.data, ccl.data]
    return HssLeaf(D)
  elseif isbranch(rcl)
    A11 = _extract_diagonal(A, rcl.left, ccl.left)
    A22 = _extract_diagonal(A, rcl.right, ccl.right)
    B12 = Matrix{Float64}(undef,0,0)
    B21 = Matrix{Float64}(undef,0,0)
    return HssNode(A11, A22, B12, B21)
  else
    throw(ErrorException("Encountered node with only one child. Make sure that each node in the binary tree has either two children or is a leaf."))
  end
end

# this function compresses given the sampling matrix of rank k
function _randcompress!(hssA::HssLeaf, A, Scol::Matrix, Srow::Matrix, Ωcol::Matrix, Ωrow::Matrix, ro::Int, co::Int, atol::Float64, rtol::Float64; rootnode=false)
  Scol = Scol - hssA.D * Ωcol
  Srow = Srow - hssA.D' * Ωrow
  # take care of column-space

  Xcol, Jcol = _interpolate(Scol', atol, rtol)
  hssA.U = Xcol'
  Scol = Scol[Jcol, :]
  U = Xcol'
  Jcol .= ro .+ Jcol

  # same for the row-space
  Xrow, Jrow = _interpolate(Srow', atol, rtol)
  hssA.V = Xrow'
  Srow = Srow[Jrow, :]
  V = Xrow'
  Jrow .= co .+ Jrow

  return hssA, Scol, Srow, Ωcol, Ωrow, Jcol, Jrow, U, V 
end
function _randcompress!(hssA::HssNode, A, Scol::Matrix, Srow::Matrix, Ωcol::Matrix, Ωrow::Matrix, ro::Int, co::Int, atol::Float64, rtol::Float64; rootnode=false)
  m1, n1 = hssA.sz1; m2, n2 = hssA.sz2
  hssA.A11, Scol1, Srow1, Ωcol1, Ωrow1, Jcol1, Jrow1, U1, V1 = _randcompress!(hssA.A11, A, Scol[1:m1, :], Srow[1:n1, :], Ωcol[1:n1, :], Ωrow[1:m1, :], ro, co, atol, rtol)
  hssA.A22, Scol2, Srow2, Ωcol2, Ωrow2, Jcol2, Jrow2, U2, V2 = _randcompress!(hssA.A22, A, Scol[m1+1:end, :], Srow[n1+1:end, :], Ωcol[n1+1:end, :], Ωrow[m1+1:end, :], ro+m1, co+n1, atol, rtol)
  
  # update the sampling matrix based on the extracted generators
  Ωcol2 = V2' * Ωcol2
  Ωcol1 = V1' * Ωcol1
  Ωrow2 = U2' * Ωrow2
  Ωrow1 = U1' * Ωrow1
  # step 1 in the algorithm: look only at extracted rows/cols
  Jcol = [Jcol1; Jcol2]
  Jrow = [Jrow1; Jrow2]
  Ωcol = [Ωcol1; Ωcol2]
  Ωrow = [Ωrow1; Ωrow2]
  # extract the correct generator blocks
  hssA.B12 = A[Jcol1, Jrow2]
  hssA.B21 = A[Jcol2, Jrow1]
  # subtract the diagonal block
  Scol = [Scol1 .- hssA.B12  * Ωcol2;  Scol2 .- hssA.B21  * Ωcol1 ];
  Srow = [Srow1 .- hssA.B21' * Ωrow2;  Srow2 .- hssA.B12' * Ωrow1 ];

  if rootnode
    rk1, wk1 = gensize(hssA.A11)
    rk2, wk2 = gensize(hssA.A22)
    hssA.R1 = Matrix{eltype(hssA)}(undef, rk1, 0)
    hssA.R2 = Matrix{eltype(hssA)}(undef, rk2, 0)
    hssA.W1 = Matrix{eltype(hssA)}(undef, wk1, 0)
    hssA.W2 = Matrix{eltype(hssA)}(undef, wk2, 0)

    return hssA
  else
    # take care of the columns
    Xcol, Jcolloc = _interpolate(Scol', atol, rtol)
    hssA.R1 = Xcol[:, 1:size(Scol1, 1)]'
    hssA.R2 = Xcol[:, size(Scol1, 1)+1:end]'
    Scol = Scol[Jcolloc, :]
    Jcol = Jcol[Jcolloc]
    U = [hssA.R1; hssA.R2]
    
    # take care of the rows
    Xrow, Jrowloc = _interpolate(Srow', atol, rtol)
    hssA.W1 = Xrow[:, 1:size(Srow1, 1)]'
    hssA.W2 = Xrow[:, size(Srow1, 1)+1:end]'
    Srow = Srow[Jrowloc, :]
    Jrow = Jrow[Jrowloc]
    V = [hssA.W1; hssA.W2]

    return hssA, Scol, Srow, Ωcol, Ωrow, Jcol, Jrow, U, V
  end
end

# interpolative decomposition
# still gotta figure out which qr decomposition to use
function _interpolate(A::Matrix{T}, atol::Float64, rtol::Float64) where T
  size(A,2) == 0 && return Matrix{T}(undef, 0,0), []
  #_, R, p = prrqr(A, tol; reltol=reltol)
  #rk = min(size(R)...)
  _, R, p  = qr(A, Val(true))
  tol = min( atol, rtol * abs(R[1,1]) )
  rk = sum(abs.(diag(R)) .> tol)
  J = p[1:rk];
  X = R[1:rk, 1:rk]\R[1:rk,invperm(p)];
  return X, J
end