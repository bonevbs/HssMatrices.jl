### Some basic operations for HSS matrices
# Written by Boris Bonev, Nov. 2020

## Scalar multiplication comes here

## hssrank based on the sizes of generators

## adds hssB to hssA in place
+(A::HssMatrix,B::Matrix) = +(promote(A,B)...)
+(A::Matrix,B::HssMatrix) = +(promote(A,B)...)