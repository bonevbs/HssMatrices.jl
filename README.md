# HssMatrices.jl

A Julia package for hierarchically semi-separable (HSS) matrices.

This package is currently under development, use at your own risk and stay tuned for more!

## Examples

This will be updated as I go. One can construct a `HssMatrix{T}` object from a dense matrix by calling
```Julia
using LinearAlgebra
using HssMatrices

A = [ abs(i-j) for i=-1:0.02:1, j=-1:0.02:1];
hssA = HssMatrix(A)
```
this will automatically build a cluster tree and compress the matrix accordingly. The compression tolerance and the minimum leaf size for the bisection cluster are stored in the global variables.

### Efficient matrix-vector and matrix-matrix multiplications
Of course we can then perform some arithmetic using HSS matrices:
```Julia
x = randn(size(hssA,2), 10);
println(norm(A*x - hssA*x))
```
We can also have a look at the generators and extract them via
```Julia
U1, V2 = generators(hssA, (1,2))
```
Another important information is the maximum off-diagonal rank. We can compute it using
```Julia
hssrank(hssA)
```

Stay tuned! More is in the works...
