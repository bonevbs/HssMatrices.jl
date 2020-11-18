include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using AbstractTrees

# test prrqr
U = randn(100,3);
V = randn(50,3);
A = U * V';
Q,R,p = HssMatrices.prrqr(A,1e-3);
norm(A[:,p] - Q[:,1:size(R,1)]*R)

# test basic hss functionality
A = [ abs(i-j) for i=-1:0.02:1.2, j=-1:0.02:1];

# test the simple implementation of cluster trees
m, n = size(A)
lsz = 10;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)
print_tree(rcl)

hssA = hss_compress_direct(A, rcl, ccl);
#println(typeof(hssA))