# HssMatrices.jl

# [![Build status (Github Actions)](https://github.com/bonevbs/HssMatrices.jl/workflows/CI/badge.svg)](https://github.com/bonevbs/HssMatrices.jl/actions)

# [![codecov.io](http://codecov.io/github/bonevbs/HssMatrices.jl/coverage.svg?branch=main)](http://codecov.io/github/bonevbs/HssMatrices.jl?branch=main)

A Julia package for hierarchically semi-separable (HSS) matrices.

This package is currently under development, use at your own risk and stay tuned for more!

## Examples

This will be updated as I go. One can construct a `HssMatrix{T}` object from a dense matrix by calling
```Julia
using LinearAlgebra
using HssMatrices

A = [ 1/(i-j) for i=-1:0.02:1, j=-1:0.02:1];
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
Alternatively, we can visualize the clustering and the off-diagonal ranks by calling
```Julia
plotranks(hssA)
```
![Plotranks](./img/plotranks.svg)

### Compression/Recompression
Basic arithmetic on hierarchical matrices requires frequent recompression of the matrices in order to guarantee that the matrices remain efficient. This is implemented in src/compression.jl via the `recompress!` routine. This is done via the rank-revealing QR decomposition to ensure efficiency. Note: our implementation of the rank-revealing QR decomposition is not optimized as of now!

Recompression can be done by simply calling the constructor on an `HssMatrix{T}` object, alternatively specifying a new compression tolerance:
```Julia
hssA = HssMatrix(hssA; tol=1e-3)
```
All compression is handled in the sense that individual HSS block rows and columns approximate the original matrix A such that the tolerance is below `tol` for this block. Similarly, if `reltol` is set to `true`, each of the blocks will be compressed in the sense that the individual block is well-approximated in the relative sense.

Alternatively, we can construct HSS matrices via random sampling.

It can also be useful to construct HSS matrices from specific datastructures. For instance, we can construct an HSS matrix from a low-rank matrix in the following fashion:
```Julia
lsz = 32
m, n = 500, 500
k = 10
U = randn(m, k); V = randn(n,k)
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)
hssA = lowrank2hss(U, V, rcl, ccl)
```

Stay tuned! More is in the works...

## Acknowledgements
This library was inspired by the amazing package [hm-toolbox](https://github.com/numpi/hm-toolbox) by Stefano Massei, Leonardo Robol and Daniel Kressner. If you are on Matlab, I suggest you to check it out.
