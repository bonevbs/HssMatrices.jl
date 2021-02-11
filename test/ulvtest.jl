include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using AbstractTrees
using Plots

K(x,y) = (x-y) != 0 ? 1/(x-y) : 10000.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1];
b = randn(size(A,2), 5);

# test the simple implementation of cluster trees
m, n = size(A)
lsz = 64;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)

# test ULV normally
hssA = hss_compress_direct(A, rcl, ccl);
x = ulvfactsolve(hssA, b);
xcor = A\b;
println(norm(x-xcor)/norm(xcor))

# test on schewed cluster trees
lsz = 701;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)
rcl.left.left.data = 1:700
rcl.left.right.data = 701:1001
#print_tree(rcl)
hssA = hss_compress_direct(A, rcl, ccl);
x = ulvfactsolve(hssA, b);
xcor = A\b;
println(norm(x-xcor)/norm(xcor))

# test on schewed cluster trees
lsz = 701;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)
ccl.left.left.data = 1:700
ccl.left.right.data = 701:1001
#print_tree(ccl)
hssA = hss_compress_direct(A, rcl, ccl);
x = ulvfactsolve(hssA, b);
xcor = A\b;
println(norm(x-xcor)/norm(xcor))

plot = plotranks(hssA)