include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using AbstractTrees
using Plots

# test prrqr
# U = randn(100,3);
# V = randn(50,3);
# A = U * V';
# Q,R,p = HssMatrices.prrqr(A,1e-3);
# norm(A[:,p] - Q[:,1:size(R,1)]*R)

# generate Cauchy matrix
K(x,y) = (x-y) != 0 ? 1/(x-y) : 1.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1];

# test the simple implementation of cluster trees
m, n = size(A)
lsz = 64;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)

# test compression
hssA = compress_direct(A, rcl, ccl);
@time hssA = compress_direct(A, rcl, ccl); 
println("approximation error with direct compression: ", norm(A - full(hssA)))
println("hss-rank with direct compression: ", hssrank(hssA))

# test randomized compression
hssB = compress_sampled(A, rcl, ccl);
println("approximation error with randomized compression: ", norm(A - full(hssB)))
println("hss-rank with randomized compression: ", hssrank(hssB))

# test recompression
hssB = recompress!(copy(hssA))
println("approximation error after recompression: ", norm(A - full(hssB)))

# test mat-vec
x = randn(size(A,2), 3);
println("error in the matrix-vector products: ", norm(A*x - hssA*x))

# test the ULV based solver
b = randn(size(A,2), 5);
x = ulvfactsolve(hssA, b);
@time x = ulvfactsolve(hssA, b);
xcor = A\b;
println("error in the inversion: ", norm(x-xcor)/norm(xcor))

# test HSS division
hssI = compress_direct(1.0*Matrix(I, n, n), ccl, ccl)
hssC = ldiv!(copy(hssA), hssI)
norm(full(hssC) - inv(A))/norm(inv(A))


# # test computation of generators
# U1, V2 = generators(hssA, (1,2))
# A12 = A[1:hssA.m1,hssA.n1+1:end];
# println(norm(A12 - U1*hssA.B12*V2'))
# U2, V1 = generators(hssA, (2,1))
# A21 = A[hssA.m1+1:end,1:hssA.n1];|
# println(norm(A21 - U2*hssA.B21*V1'))


# #println(typeof(hssA)) 

# test the ULV based solver
# x = ulvfactsolve(hssA, b);

# @time x = ulvfactsolve(hssA, b);

# # test plotting
# plot = plotranks(hssA)

### TODO
# clean up the library Definitions
