using OrthogonalPolynomialsQuasi, FillArrays, BandedMatrices, ContinuumArrays, ArrayLayouts, Test
import OrthogonalPolynomialsQuasi: NormalizationConstant, recurrencecoefficients, Normalized, Clenshaw
import ContinuumArrays: BasisLayout


@testset "Normalized" begin
    P = Legendre()
    Q = Normalized(P)

    @testset "Basic" begin
        @test MemoryLayout(Q) isa BasisLayout
        @test @inferred(Q\Q) ≡ Eye(∞)
    end
    
    @testset "recurrencecoefficients" begin
        A,B,C = recurrencecoefficients(Q)
        @test A[3:∞][1:10] == A[3:12]
        @test B[3:∞] ≡ Zeros(∞)
    end

    @testset "Evaluation" begin
        M = P'P
        @test Q[0.1,1] == 1/sqrt(2)
        @test Q[0.1,2] ≈ sqrt(1/M[2,2]) * P[0.1,2]
        @test Q[0.1,Base.OneTo(10)] ≈ Q[0.1,1:10] ≈ sqrt.(inv(M)[1:10,1:10]) * P[0.1,Base.OneTo(10)]
        @test (Q'Q)[1:10,1:10] ≈ I
    end

    @testset "Expansion" begin
        f = Q*[1:5; zeros(∞)]
        @test f[0.1] ≈ Q[0.1,1:5]'*(1:5) ≈ f[[0.1]][1]
        x = axes(f,1)
        w = Q * (Q \ (1 .- x.^2));
        @test w[0.1] ≈ (1-0.1^2) ≈ w[[0.1]][1]
    end

    @testset "Conversion" begin
        @test ((P \ Q) * (Q \ P))[1:10,1:10] ≈ (Q \Q)[1:10,1:10] ≈ I
        @test (Jacobi(1,1) \ Q)[1:10,1:10] ≈ ((Jacobi(1,1) \ P) * (P \ Q))[1:10,1:10]
    end

    @testset "Derivatives" begin
        D = Derivative(axes(Q,1))
        f = Q*[1:5; zeros(∞)]
        h = 0.000001
        @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=1E-4
    end

    @testset "Multiplication" begin
        x = axes(Q,1)
        @test Q \ (x .* Q) isa Symmetric

        w = P * (P \ (1 .- x.^2));
        W = Q \ (w .* Q)
        @test W isa Clenshaw
        @test bandwidths(W) == (2,2)
        W̃ = Q' * (w .* Q)
        @test W[1:10,1:10] ≈ W[1:10,1:10]' ≈ W̃[1:10,1:10]
    end
end
