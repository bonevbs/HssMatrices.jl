### This File contains all compression routines

## Direct compression
# as described in
# Xia, J., Chandrasekaran, S., Gu, M., & Li, X. S. (2010). Fast algorithms for hierarchically semiseparable matrices.
# Numerical Linear Algebra with Applications, 17(6), 953–976. https://doi.org/10.1002/nla.691

# wrapper function that will be exported
function hss_compress_direct(A::Matrix{T}, rcl::BinaryNode, ccl::BinaryNode, tol=tol; reltol=reltol) where T
  m = length(rcl.data); n = length(ccl.data)
  hssA = hss_from_cluster(rcl, ccl)
  Brow = Array{T}(undef, m, 0)
  Bcol = Array{T}(undef, 0, n)
  hss_compress_direct!(hssA, A, Brow, Bcol, 0, 0, m, n, tol; reltol)
  return hssA
end

# recursive function
function hss_compress_direct!(hssA::HssMatrix, A::Matrix{T}, Brow::Matrix{T}, Bcol::Matrix{T}, ro, co, m, n, tol; reltol) where T
  if hssA.leafnode
    hssA.D = A[ro+1:ro+m, co+1:co+n]
    # compress HSS block row/col
    hssA.U, Brow = compress_block!(Brow, tol; reltol)
    hssA.V, Bcol = compress_block!(copy(Bcol'), tol; reltol); Bcol = copy(Bcol') # quick fix to avoid Julia's lazy transpose
  else
    m1 = hssA.m1; n1 = hssA.n1; m2 = hssA.m2; n2 = hssA.n2

    # form blocks to be compressed in the children step
    Brow1 = hcat( A[ro+1:ro+m1, co+n1+1:co+n1+n2], Brow[1:m1, :] ) #FIXME: rootnode?
    Bcol1 = vcat( A[ro+m1+1:ro+m1+m2, co+1:co+n1], Bcol[:, 1:n1] )
    Brow1, Bcol1 = hss_compress_direct!(hssA.A11, A, Brow1, Bcol1, ro, co, m1, n1, tol; reltol)
    
    # form blocks to be compressed in the children stepnull
    Brow2 = hcat( Bcol1[1:m2, :] , Brow[m1+1:end, :] )
    Bcol2 = vcat( Brow1[:, 1:n2] , Bcol[:, n1+1:end] )
    Brow2, Bcol2 = hss_compress_direct!(hssA.A22, A, Brow2, Bcol2, ro+m1, co+n1, m2, n2, tol; reltol)

    # figure out dimensions of reduced blocks
    rm1 = size(Brow1,1); rn1 = size(Bcol1, 2)
    rm2 = size(Brow2,1); rn2 = size(Bcol2, 2)

    hssA.B12 = Bcol2[1:rm1, :]
    hssA.B21 = Brow2[:, 1:rn1]

    if !hssA.rootnode
      # clean up stuff from the front and form the composed HSS block row/col for compression
      Brow = vcat( Brow1[:, n2+1:end],  Brow2[:, rn1+1:end])
      Bcol = hcat( Bcol1[m2+1:end, :],  Bcol2[rm1+1:end, :])

      # do the actual compression and disentangle blocks of the translation operators
      R, Brow = compress_block!(Brow, tol; reltol)
      hssA.R1 = R[1:rm1, :]
      hssA.R2 = R[rm1+1:end, :]
    
      W, Bcol = compress_block!(copy(Bcol'), tol; reltol); Bcol = copy(Bcol')
      hssA.W1 = W[1:rn1, :]
      hssA.W2 = W[rn1+1:end, :]
    end
  end
  return Brow, Bcol
end

function compress_block!(A::Matrix{T}, tol; reltol) where T
  Q, R, p = prrqr!(A, tol; reltol)
  rk = size(R,1);
  return Q[:,1:rk], R[1:rk, invperm(p)]
end

