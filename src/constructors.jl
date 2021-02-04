### Special constructors for HSS matrices
#
# Written by Boris Bonev, Feb. 2020

function lowrank2hss(U::Matrix{T}, V::Matrix{T}, rcl::ClusterTree, ccl::ClusterTree) where T
  (k = size(U,2)) == size(V,2) || throw(ArgumentError("second dimension of U and V must agree"))
  return _lowrank2hss(U, V, rcl, ccl, k)
end

function _lowrank2hss(U::Matrix{T}, V::Matrix{T}, rcl::ClusterTree, ccl::ClusterTree, k::Int) where T
  if isleaf(rcl) && isleaf(ccl)
    HssLeaf(U[rcl.data,:]*V[ccl.data,:]', U[rcl.data,:], V[ccl.data,:])
  elseif isbranch(rcl) && isbranch(ccl)
    A11 = _lowrank2hss(U, V, rcl.left, ccl.left, k)
    A22 = _lowrank2hss(U, V, rcl.right, ccl.right, k)
    HssNode(A11, A22, Matrix{T}(I,k,k), Matrix{T}(I,k,k), Matrix{T}(I,k,k), Matrix{T}(I,k,k), Matrix{T}(I,k,k), Matrix{T}(I,k,k))
  else
    throw(ArgumentError("row and column clusters are not compatible"))
  end
end