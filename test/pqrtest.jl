include("../src/prrqr2.jl")
using BenchmarkTools
using LinearAlgebra
using LowRankApprox
using Random

Random.seed!(0)

tol = 1e-3

n = 10; k = 10; m = 100
s = zeros(n)
s[1:k] = @. exp(-(1:k));
U, _ = qr(randn(n,k))
V, _ = qr(randn(m,k))
A = U*diagm(s)*V';


#println("Benchmarking rank-revealing QR...")
#@btime _ = prrqr!(randn(n,100), tol; reltol=true)

println("Testing accuracy...")
A = U*diagm(s)*V';
X = copy(A)
Q, R = _compress_block!(X; tol, reltol=false)
X = copy(A)
@time Q, R = _compress_block!(X; tol, reltol=false)
println("error in the truncated QR decomposition: ", norm(A - Q*R)/norm(A))


#_, σ, _ = svd(A);
#plot(1:n, σ, yaxis=:log, label="singular values") 