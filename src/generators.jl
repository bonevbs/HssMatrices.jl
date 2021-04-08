### Functions concerning the computation and modification of generators
#
# Re-Written by Boris Bonev, Jan. 2021

## Convenience access functions to construct off-diagonal blocks and tall generators
function generators(hssA::HssMatrix{T}, ul::Tuple{S,S}) where {T, S <: Integer}
  if ul == (1,1)
    U, V = generators(hssA.A11)
  elseif ul == (1,2)
    U, _ = generators(hssA.A11)
    _, V = generators(hssA.A22)
  elseif ul == (2,1)
    _, V = generators(hssA.A11)
    U, _ = generators(hssA.A22)
  elseif ul == (2,2)
    U, V = generators(hssA.A22)
  else
    error("cannot make sense of ul")
  end
  return U, V
end

# recursive function to compute the generators corresponding to this subblock
function generators(hssA::HssMatrix)
  rootnode(hssA) && error("Ambiguous call to generators on rootnode")
  if isleaf(hssA)
    return hssA.U, hssA.V
  else
    U1, V1 = generators(hssA.A11)
    U2, V2 = generators(hssA.A22)
    return [U1*hssA.R1; U2*hssA.R2], [V1*hssA.W1; V2*hssA.W2]
  end
end

# recursive function for getting just the desired index of colmn/row generators
function _getindex_colgenerator(hssA::HssMatrix, i::Int)
  if isleaf(hssA)
    return hssA.U[i,:]'
  else
    m1 = hssA.sz1[1]
    if i <= m1
      return _getindex_colgenerator(hssA.A11, i)*hssA.R1
    else
      return _getindex_colgenerator(hssA.A22, i-m1)*hssA.R2
    end
  end
end
function _getindex_rowgenerator(hssA::HssMatrix, j::Int)
  if isleaf(hssA)
    hssA.V[j,:]'
  else
    n1 = hssA.sz1[2]
    if j <= n1
      return _getindex_rowgenerator(hssA.A11, j)*hssA.W1
    else
      return _getindex_rowgenerator(hssA.A22, j-n1)*hssA.W2
    end
  end
end

#offdiag(hssA::HssNode, ::Val{:upper}) = generators(hssA.A11)[1]*hssA.B12*generators(hssA.A11)[2]'

## orthogonalize generators
function orthonormalize_generators!(hssA::HssMatrix{T}) where T
  if isleaf(hssA)
    U1 = pqrfact!(hssA.A11.U, sketch=:none); hssA.U = Matrix(U1.Q)
    V1 = pqrfact!(hssA.A11.V, sketch=:none); hssA.V = Matrix(V1.Q)
  else
    if isleaf(hssA.A11)
      U1 = pqrfact!(hssA.A11.U, sketch=:none); hssA.A11.U = Matrix(U1.Q)
      V1 = pqrfact!(hssA.A11.V, sketch=:none); hssA.A11.V = Matrix(V1.Q)
    else
      hssA.A11 = orthonormalize_generators!(hssA.A11)
      U1 = pqrfact!([hssA.A11.R1; hssA.A11.R2], sketch=:none)
      V1 = pqrfact!([hssA.A11.W1; hssA.A11.W2], sketch=:none)
      rm1 = size(hssA.A11.R1, 1)
      R = Matrix(U1.Q)
      hssA.A11.R1 = R[1:rm1,:]
      hssA.A11.R2 = R[rm1+1:end,:]
      rn1 = size(hssA.A11.W1, 1)
      W = Matrix(V1.Q)
      hssA.A11.W1 = W[1:rn1,:]
      hssA.A11.W2 = W[rn1+1:end,:]
    end

    if isleaf(hssA.A22)
      U2 = pqrfact!(hssA.A22.U, sketch=:none); hssA.A22.U = Matrix(U2.Q)
      V2 = pqrfact!(hssA.A22.V, sketch=:none); hssA.A22.V = Matrix(V2.Q)
    else
      hssA.A22 = orthonormalize_generators!(hssA.A22)
      U2 = pqrfact!([hssA.A22.R1; hssA.A22.R2], sketch=:none)
      V2 = pqrfact!([hssA.A22.W1; hssA.A22.W2], sketch=:none)
      rm1 = size(hssA.A22.R1, 1)
      R = Matrix(U2.Q)
      hssA.A22.R1 = R[1:rm1,:]
      hssA.A22.R2 = R[rm1+1:end,:]
      rn1 = size(hssA.A22.W1, 1)
      W = Matrix(V2.Q)
      hssA.A22.W1 = W[1:rn1,:]
      hssA.A22.W2 = W[rn1+1:end,:]
    end
    ipU1 = invperm(U1.p); ipV1 = invperm(V1.p)
    ipU2 = invperm(U2.p); ipV2 = invperm(V2.p)

    hssA.B12 = U1.R[:, ipU1]*hssA.B12*V2.R[:, ipV2]'
    hssA.B21 = U2.R[:, ipU2]*hssA.B21*V1.R[:, ipV1]'

    if !isroot(hssA)
      hssA.R1 = U1.R[:, ipU1]*hssA.R1
      hssA.R2 = U2.R[:, ipU2]*hssA.R2
      hssA.W1 = V1.R[:, ipV1]*hssA.W1
      hssA.W2 = V2.R[:, ipV2]*hssA.W2
    end
  end
  return hssA
end