### Some basic operations for HSS matrices
# Written by Boris Bonev, Nov. 2020

## Scalar multiplication comes here

## hssrank based on the sizes of generators
function hssrank(hssA::HssMatrix{T}) where T
  if hssA.leafnode
    rk = 0
  else
    rk = max(hssrank(hssA.A11), hssrank(hssA.A22), rank(hssA.B12), rank(hssA.B21))
  end
  return rk
end

## adds hssB to hssA in place
+(A::HssMatrix,B::Matrix) = +(promote(A,B)...)
+(A::Matrix,B::HssMatrix) = +(promote(A,B)...)