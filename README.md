# HssMatrices.jl

A Julia package for hierarchically semi-separable (HSS) matrices.

This package is currently under development, use at your own risk and stay tuned for more!

## Examples

This will be updated as I go. One can construct a `HssMatrix{T}` object from a dense matrix by calling
```
using LinearAlgebra
using HssMatrices

A = randn(1024,1024)
hssA = HssMatrix(A)
```
this will automatically build a cluster tree and compress the matrix accordingly. The compression tolerance and the minimum leaf size for the bisection cluster are stored in the global variables.
