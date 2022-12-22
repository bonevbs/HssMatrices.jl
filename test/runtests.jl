using Test, LinearAlgebra, HssMatrices

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

@testset for T in [Float32, Float64, ComplexF32,ComplexF64]

    # generate Cauchy matrix
    K(x,y) = (x-y) > 0 ? 0.001/(x-y) : 2.
    A = [ T(K(x,y)) for x=-1:0.001:1, y=-1:0.001:1];
    if T <: Complex
        A = A + 1im .* A
    end
    m, n = size(A)
    U = randn(T,n,3); V = randn(T,n,3)
    # "safety" factor
    c = 50.
    tol = 1E-6
    HssMatrices.setopts(atol=tol)
    HssMatrices.setopts(rtol=tol)

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
        hssC = lowrank2hss(U, V, ccl, ccl)
        @test norm(U*V' - full(hssC))/norm(U*V') ≤ c*tol
        @test hssrank(hssC) == 3
    end;

    @testset "random access" begin
        I = rand(1:size(A,1), 10)
        J = rand(1:size(A,1), 10)
        function testAccuracy(expected, result)
            @test norm(expected - result)/norm(expected) ≤ c*HssOptions().rtol || norm(expected - result) ≤ c*HssOptions().atol
        end
        testAccuracy(A[I,J], compress(A, rcl, ccl)[I,J])
        testAccuracy(A[I,:], compress(A, rcl, ccl)[I,:])
        testAccuracy(A[:,J], compress(A, rcl, ccl)[:,J])
    end;

    @testset "arithmetic" begin
        hssA = compress(A, rcl, ccl);
        hssC = lowrank2hss(U, V, ccl, ccl)
        @test norm(A' - full(hssA'))/norm(A) ≤ c*HssOptions().rtol || norm(A' - full(hssA')) ≤ c*HssOptions().atol
        x = randn(T, n, 5);
        @test norm(A*x - hssA*x)/norm(A*x) ≤ c*HssOptions().rtol || norm(A*x - hssA*x) ≤ c*HssOptions().atol
        @test norm(full(hssA*hssC) - (A*U)*V')/norm((A*U)*V') ≤ c*HssOptions().rtol || norm(A*x - hssA*x) ≤ c*HssOptions().atol
        rhs = randn(T,n, 5); x = hssA\rhs; x0 = A\rhs;
        @test norm(x0 - x)/norm(x0) ≤ c*HssOptions().rtol || norm(x0 - x) ≤ c*HssOptions().atol
        Id(i,j) = Matrix{T}(i.*ones(length(j))' .== ones(length(i)).*j')
        IdOp = LinearMap{T}(n, n, (y,_,x) -> x, (y,_,x) -> x, (i,j) -> Id(i,j))
        hssI = randcompress(IdOp, ccl, ccl, 0)
        @test norm(full(hssA*hssI) - full(hssA))/norm(full(hssA)) ≤ c*tol
        Ainv = inv(A)
        @test norm(Ainv - full(hssA\hssI))/norm(Ainv) ≤ c*HssOptions().rtol || norm(Ainv - full(hssA\hssI)) ≤ c*HssOptions().atol
        @test norm(Ainv - full(hssI/hssA))/norm(Ainv) ≤ c*HssOptions().rtol || norm(Ainv - full(hssI/hssA)) ≤ c*HssOptions().atol
        hssA.A11 = prune_leaves!(hssA.A11)
        hssI.A11 = prune_leaves!(hssI.A11)
        @test norm(Ainv - full(hssA\hssI))/norm(Ainv) ≤ c*HssOptions().rtol || norm(Ainv - full(hssA\hssI)) ≤ c*HssOptions().atol
        @test norm(Ainv - full(hssI/hssA))/norm(Ainv) ≤ c*HssOptions().rtol || norm(Ainv - full(hssI/hssA)) ≤ c*HssOptions().atol
    end
end 