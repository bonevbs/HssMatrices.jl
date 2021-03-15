# HssMatrices.jl

`HssMatrices` is a Julia package for hierarchically semi-separable (HSS) matrices. These matrices are a type of hierarchically structured matrices, which often arise in the context of solving PDEs numerically. This package is intendend to help users experiment with these matrices and algorithms/arithmetic related to these matrices. It implements compression routines, arithmetic as well as helpful routines for clustering and visualization.

## Getting started

Let us generate a simple Kernel matrix and convert it into HSS format:
```Julia
using LinearAlgebra
using HssMatrices

K(x,y) = (x-y) != 0 ? 1/(x-y) : 1.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1]
hssA = hss(A)
```
This will automatically build a cluster tree and compress the matrix accordingly. `hss()` acts as a smart constructor, which will construct the matrix depending on the supplied matrix and parameters. We can either pass these parameters, for instance by doing:
```Julia
hssA = hss(A, leafsize=64, atol=1e-6, rtol=1e-6)
```


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
This library was inspired by the amazing package [hm-toolbox](https://github.com/numpi/hm-toolbox) by Stefano Massei, Leonardo Robol and Daniel Kressner. If you are using Matlab, I highly recommend to try this package.

In numerous occasions, members of the Julia Slack channel have helped me with the challenges of writing my first library in Julia. I would like to acknowledge their support.
