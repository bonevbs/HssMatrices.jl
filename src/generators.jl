### Convenience access functions to construct off-diagonal blocks and tall generators
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

# recursive function to compute the generators
function generators(hssA::HssMatrix{T}) where T
  if hssA.leafnode
    return hssA.U, hssA.V
  else
    U1, V1 = generators(hssA.A11)
    U2, V2 = generators(hssA.A22)
    U = [ U1*hssA.R1; U2*hssA.R2 ]
    V = [ V1*hssA.W1; V2*hssA.W2 ]
    return U, V
  end
end