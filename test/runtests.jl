using Test, LinearAlgebra, HssMatrices

# generate Cauchy matrix
K(x,y) = (x-y) > 0 ? 0.001/(x-y) : 2.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1];
m, n = size(A)
# "safety" factor
c = 1000.0

@testset "options" begin
    HssMatrices.setopts(atol=1e-6)
    @test HssOptions().atol == 1e-6
    HssMatrices.setopts(rtol=1e-6)
    @test HssOptions().rtol == 1e-6
    HssMatrices.setopts(leafsize=50)
    @test HssOptions().leafsize == 50
    HssMatrices.setopts(noversampling=5)
    @test HssOptions().noversampling == 5
    HssMatrices.setopts(stepsize=10)
    @test HssOptions().stepsize == 10
end

rcl = bisection_cluster(1:m)
ccl = bisection_cluster(1:n)

@testset "compression" begin
    hssA = compress(A, rcl, ccl);
    @test norm(A - full(hssA))/norm(A) ≤ c*HssOptions().rtol || norm(A - full(hssA)) ≤ c*HssOptions().atol
    hssB = randcompress_adaptive(A, rcl, ccl); rk = hssrank(hssB)
    @test norm(A - full(hssB))/norm(A) ≤ c*HssOptions().rtol || norm(A - full(hssB)) ≤ c*HssOptions().atol
    hssB = recompress!(hssB)
    @test norm(A - full(hssB))/norm(A) ≤ c*HssOptions().rtol || norm(A - full(hssB)) ≤ c*HssOptions().atol
    @test hssrank(hssB) ≤ rk
end;

@testset "arithmetic" begin
    hssA = compress(A, rcl, ccl);
    @test norm(A' - full(hssA'))/norm(A) ≤ c*HssOptions().rtol || norm(A' - full(hssA')) ≤ c*HssOptions().atol
    x = randn(n, 5);
    @test norm(A*x - hssA*x)/norm(A*x) ≤ c*HssOptions().rtol || norm(A*x - hssA*x) ≤ c*HssOptions().atol
    rhs = randn(n, 5); x = hssA\rhs; x0 = A\rhs;
    @test norm(x0 - x)/norm(x0) ≤ c*HssOptions().rtol || norm(x0 - x) ≤ c*HssOptions().atol
    Id(i,j) = Matrix{Float64}(i.*ones(length(j))' .== ones(length(i)).*j')
    IdOp = LinearMap{Float64}(n, n, (y,_,x) -> x, (y,_,x) -> x, (i,j) -> Id(i,j), nothing)
    hssI = randcompress(IdOp, ccl, ccl, 0)
    @test norm(full(hssA*hssI) - full(hssA))/norm(full(hssA)) ≤ c*eps()
    Ainv = inv(A)
    @test norm(Ainv - full(hssA\hssI))/norm(Ainv) ≤ c*HssOptions().rtol || norm(Ainv - full(hssA\hssI)) ≤ c*HssOptions().atol
    @test norm(Ainv - full(hssI/hssA))/norm(Ainv) ≤ c*HssOptions().rtol || norm(Ainv - full(hssI/hssA)) ≤ c*HssOptions().atol
end