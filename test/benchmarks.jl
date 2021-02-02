include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using BenchmarkTools

### run benchmarks on Cauchy matrix
K(x,y) = (x-y) != 0 ? 1/(x-y) : 10000.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1];
b = randn(size(A,2), 5);

# test the simple implementation of cluster trees
m, n = size(A)
#lsz = 64;
lsz = 64;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)

hssA = hss_compress_direct(A, rcl, ccl);

# time compression
println("Benchmarking compression...")
@btime hssA = hss_compress_direct(A, rcl, ccl);

# time matvec
println("Benchmarking matvec...")
x = randn(size(A,2), 10);
@btime y = hssA*x

# time ulvfactsolve
println("Benchmarking ulvfactsolve...")
b = randn(size(A,2), 10);
@btime x = ulvfactsolve(hssA, b);
