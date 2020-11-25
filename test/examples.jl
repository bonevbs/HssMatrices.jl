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
A = [ abs(i-j) for i=-1:0.02:1, j=-1:0.02:1];

# test the simple implementation of cluster trees
m, n = size(A)
lsz = 10;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)
# print_tree(rcl)

# test compression
hssA = hss_compress_direct(A, rcl, ccl);

# test computation of generators
U1, V2 = generators(hssA, (1,2))
A12 = A[1:hssA.m1,hssA.n1+1:end];
println(norm(A12 - U1*hssA.B12*V2'))
U2, V1 = generators(hssA, (2,1))
A21 = A[hssA.m1+1:end,1:hssA.n1];
println(norm(A21 - U2*hssA.B21*V1'))

# test mat-vec
x = randn(101, 2);
println(norm(A*x - hssA*x));

#println(typeof(hssA)) 

# test orthonormalization
m, n = size(hssA.A11.A11.R1);
hssA.A11.A11.R1 = randn(m, n);
A = Matrix(hssA);
orthonormalize_generators!(hssA)
println(norm(A - Matrix(hssA)))

# test recompression
println("approximation error before recompression: ", norm(A - Matrix(A)))
hss_recompress!(hssA,1e-1; reltol=false)
println("approximation error after recompression: ", norm(A - Matrix(A)))

# test plotting
plotranks(hssA)