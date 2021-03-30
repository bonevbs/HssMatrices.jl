include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using AbstractTrees
using Plots
using BenchmarkTools

n = 1000
K(x,y) = (x-y) != 0 ? 1/(x-y) : 1000.
A = [ K(x,y) for x=0:1/(n-1):1, y=0:1/(n-1):1];
b = randn(n, 1);

m, n = size(A)
lsz = 64;
rcl = bisection_cluster(1:m, leafsize=lsz)
ccl = bisection_cluster(1:n, leafsize=lsz)
hssA = compress(A, rcl, ccl);

hssA\hssA

n = 20000
K(x,y) = (x-y) != 0 ? 1/(x-y) : 1000.
A = [ K(x,y) for x=0:1/(n-1):1, y=0:1/(n-1):1];
b = randn(n, 1);

# test the simple implementation of cluster trees
m, n = size(A)
lsz = 64;
rcl = bisection_cluster(1:m, leafsize=lsz)
ccl = bisection_cluster(1:n, leafsize=lsz)
hssA = compress(A, rcl, ccl);

@profview hssA\hssA

# # test ULV normally
# x = ulvfactsolve(hssA, b);
# xcor = A\b;
# println(norm(x-xcor)/norm(xcor))

# # test on schewed cluster trees
# lsz = 701;
# rcl = bisection_cluster(1:m, leafsize=lsz)
# ccl = bisection_cluster(1:n, leafsize=lsz)
# rcl.left.left.data = 1:700
# rcl.left.right.data = 701:1001
# #print_tree(rcl)
# hssA = compress(A, rcl, ccl);
# x = ulvfactsolve(hssA, b);
# xcor = A\b;
# println(norm(x-xcor)/norm(xcor))  

# # test on schewed cluster trees
# lsz = 701;
# rcl = bisection_cluster(1:m, leafsize=lsz)
# ccl = bisection_cluster(1:n, leafsize=lsz)
# ccl.left.left.data = 1:700
# ccl.left.right.data = 701:1001
# #print_tree(ccl)
# hssA = compress(A, rcl, ccl);
# x = ulvfactsolve(hssA, b);
# xcor = A\b;
# println(norm(x-xcor)/norm(xcor))

plot = plotranks(hssA)