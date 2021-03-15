include("../src/prrqr2.jl")
using BenchmarkTools
using LinearAlgebra
using LowRankApprox
using Random

Random.seed!(0)

tol = 1e-3

n = 1000; k = 10; m = 1000
s = zeros(k)
s[1:k] = @. exp(-(1:k));
U = randn(n,k)
V = randn(m,k)
A = U*diagm(s)*V';


#println("Benchmarking rank-revealing QR...")
#@btime _ = prrqr!(randn(n,100), tol; reltol=true)

println("Testing custom implementation...")
X = copy(A)
Q, R = _compress_block!(X; tol, reltol=false)
X = copy(A)
@time Q, R = _compress_block!(X; tol, reltol=false)
println("error in the truncated QR decomposition: ", norm(A - Q*R)/norm(A))

println("Testing LowRankApprox.jl implementation...")
# LowRankApprox.jl implementation
X = copy(A)
F = pqrfact(X; atol = tol)
X = copy(A)
@time F = pqrfact(X; atol = tol)
println("error in the LowRankApprox.jl PQR decomposition: ", norm(A - F.Q*F.R)/norm(A))


#_, σ, _ = svd(A);
#plot(1:n, σ, yaxis=:log, label="singular values") 