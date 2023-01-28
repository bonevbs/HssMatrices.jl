include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using AbstractTrees
using Plots

# generate Cauchy matrix
K(x,y) = (x-y) > 0. ? 0.001/(x-y) : 2.
#K(x,y) = (x-y) != 0 ? 1/(x-y) : 1.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1];
C = inv(A);

# test the simple implementation of cluster trees
m, n = size(A)
HssMatrices.setopts(leafsize=64)
rcl = bisection_cluster(1:m)
ccl = bisection_cluster(1:n)

# test compression
hssA = compress(A, rcl, ccl);
println("rel. approximation error with direct compression: ", norm(A - full(hssA))/norm(A))
println("abs. approximation error with direct compression: ", norm(A - full(hssA)))
println("hss-rank with direct compression: ", hssrank(hssA))

# test randomized compression
hssB = randcompress_adaptive(A, rcl, ccl);
println("rel. approximation error with randomized compression: ", norm(A - full(hssB))/norm(A))
println("abs. approximation error with randomized compression: ", norm(A - full(hssB)))
println("hss-rank with randomized compression: ", hssrank(hssB))

# test recompression
hssB = recompress!(copy(hssA))
println("rel. approximation error after recompression: ", norm(A - full(hssB))/norm(A))
println("abs. approximation error after recompression: ", norm(A - full(hssB)))
println("hss-rank after recompression: ", hssrank(hssB))

# test mat-vec
x = randn(size(A,2), 3);
println("rel. error in the matrix-vector products: ", norm(A*x - hssA*x)/norm(A*x))
println("abs. error in the matrix-vector products: ", norm(A*x - hssA*x))

# test the ULV based solver
b = randn(size(A,2), 5);
x = ulvfactsolve(hssA, b);
xcor = A\b;
println("rel. error in the solution of Ax = b: ", norm(x-xcor)/norm(xcor))
println("abs. error in the solution of Ax = b: ", norm(x-xcor))

# test left HSS division
Id(i,j) = Matrix{Float64}(i.*ones(length(j))' .== ones(length(i)).*j')
IdOp = LinearMap{Float64}(n, n, (y,_,x) -> x, (y,_,x) -> x, (i,j) -> Id(i,j))
hssI = randcompress(IdOp, ccl, ccl, 0)
hssC = ldiv!(hssA, hssI)
println("rel. error in the solution of AX = I: ", norm(full(hssC) - C) / norm(C) )
println("abs. error in the solution of AX = I: ", norm(full(hssC) - C) )

# test left HSS division
Id(i,j) = Matrix{Float64}(i.*ones(length(j))' .== ones(length(i)).*j')
IdOp = LinearMap{Float64}(n, n, (y,_,x) -> x, (y,_,x) -> x, (i,j) -> Id(i,j))
hssI = randcompress(IdOp, ccl, ccl, 0)
hssC = rdiv!(hssI, hssA)
println("rel. error in the solution of XA = I: ", norm(full(hssC) - C) / norm(C) )
println("abs. error in the solution of XA = I: ", norm(full(hssC) - C) )

# desequilibrate cluster trees and try again
hssA.A11 = prune_leaves!(hssA.A11)
hssI = randcompress(IdOp, ccl, ccl, 0)
hssI.A11 = prune_leaves!(hssI.A11)
hssC = ldiv!(hssA, hssI)
println("rel. error in the solution of XA = I: ", norm(full(hssC) - C) / norm(C) )
println("abs. error in the solution of XA = I: ", norm(full(hssC) - C) )


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