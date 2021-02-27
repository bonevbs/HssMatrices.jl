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
generators(hssA::HssLeaf) = hssA.U, hssA.V
function generators(hssA::HssNode)
  U1, V1 = generators(hssA.A11)
  U2, V2 = generators(hssA.A22)
  return [U1*hssA.R1; U2*hssA.R2], [V1*hssA.W1; V2*hssA.W2]
end

#offdiag(hssA::HssNode, ::Val{:upper}) = generators(hssA.A11)[1]*hssA.B12*generators(hssA.A11)[2]'

## orthogonalize generators
function orthonormalize_generators!(hssA::HssLeaf{T}) where T
  U1 = qr(hssA.U); hssA.U = Matrix(U1.Q)
  V1 = qr(hssA.V); hssA.V = Matrix(V1.Q)
  return hssA
end

function orthonormalize_generators!(hssA::HssNode{T}) where T
  if isleaf(hssA.A11)
    U1 = qr(hssA.A11.U); hssA.A11.U = Matrix(U1.Q)
    V1 = qr(hssA.A11.V); hssA.A11.V = Matrix(V1.Q)
  else
    orthonormalize_generators!(hssA.A11)
    U1 = qr([hssA.A11.R1; hssA.A11.R2])
    V1 = qr([hssA.A11.W1; hssA.A11.W2])
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
    U2 = qr(hssA.A22.U); hssA.A22.U = Matrix(U2.Q)
    V2 = qr(hssA.A22.V); hssA.A22.V = Matrix(V2.Q)
  else
    orthonormalize_generators!(hssA.A22)
    U2 = qr([hssA.A22.R1; hssA.A22.R2])
    V2 = qr([hssA.A22.W1; hssA.A22.W2])
    rm1 = size(hssA.A22.R1, 1)
    R = Matrix(U2.Q)
    hssA.A22.R1 = R[1:rm1,:]
    hssA.A22.R2 = R[rm1+1:end,:]
    rn1 = size(hssA.A22.W1, 1)
    W = Matrix(V2.Q)
    hssA.A22.W1 = W[1:rn1,:]
    hssA.A22.W2 = W[rn1+1:end,:]
  end

  hssA.B12 = U1.R*hssA.B12*V2.R'
  hssA.B21 = U2.R*hssA.B21*V1.R'

  hssA.R1 = U1.R*hssA.R1
  hssA.R2 = U2.R*hssA.R2
  hssA.W1 = V1.R*hssA.W1
  hssA.W2 = V2.R*hssA.W2

  return hssA
end