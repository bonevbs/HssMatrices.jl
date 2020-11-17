### This File contains all compression routines
#include("./cluster_trees.jl")
#include("./prrqr.jl")
using LinearAlgebra
using InvertedIndices
using DataStructures

## direct compression
# wrapper function that will be exported
function hss_compress_direct(A::Matrix{T}, rcl::BinaryNode, ccl::BinaryNode, tol=1e-9; reltol=true) where T
  m = length(rcl.data); n = length(ccl.data)
  hssA = hss_from_cluster(rcl, ccl)
  hss_compress_direct!(hssA, A, 0, 0, m, n, tol; reltol)
  return hssA
end

# recursive function
function hss_compress_direct!(hssA::HssMatrix, A::Matrix{T}, ro, co, m, n, tol; reltol=true) where T
  if hssA.leafnode
    hssA.D = A[ro+1:ro+m, co+1:co+n]

    # compress HSS block row
    Brow = A[ro+1:ro+m, Not(co+1:co+n)]
    hssA.U, Brow = compress_block(Brow, tol; reltol)

    # same for the block column
    Bcol = copy(A[Not(ro+1:ro+m), co+1:co+n]') # some ugly copying going on as there is no routine for the adjoint yet
    hssA.V, Bcol = compress_block(Bcol, tol; reltol)
  else
    m1 = hssA.m1; n1 = hssA.n1; m2 = hssA.m2; n2 = hssA.n2
    # deal with children nodes
    Brow1, Bcol1 = hss_compress_direct!(hssA.A11, A, ro, co, m1, n1, tol; reltol)
    Brow2, Bcol2 = hss_compress_direct!(hssA.A22, A, ro+m1, co+n1, m2, n2, tol; reltol)

    Brow = [ Brow1[:, Not(co+1:co+n2)]; Brow2[:, Not(co+1:co+n1)] ]
    R, Brow = compress_block(Brow, tol; reltol)
    

    Bcol = [ Bcol1[:, Not(ro+1:ro+m2)]; Bcol2[:, Not(ro+1:ro+m1)] ]
    W, Bcol = compress_block(Bcol, tol; reltol)
  end
  return Brow, Bcol
end

function compress_block(A::Matrix{T}, tol; reltol=true) where T
  Q, R, p = prrqr(A, tol; reltol)
  rk = size(R,1);
  return Q[:,1:rk], R[1:rk, invperm(p)]
end

