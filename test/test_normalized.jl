using OrthogonalPolynomialsQuasi, FillArrays, BandedMatrices, ContinuumArrays, ArrayLayouts, Base64, Test
import OrthogonalPolynomialsQuasi: NormalizationConstant, recurrencecoefficients, Normalized, Clenshaw, PaddedLayout, weighted
import ContinuumArrays: BasisLayout


@testset "Normalized" begin
    @testset "Legendre" begin
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
            @test MemoryLayout(Q \ (1 .- x.^2)) isa PaddedLayout
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

            P = Normalized(Legendre())
            x = axes(P,1)
            w = x .+ x.^2 .+ 1 # w[x] == exp(x)
            W = P \ (w .* P)
            @test W isa Clenshaw
        end

        @testset "show" begin
            @test stringmime("text/plain", Normalized(Legendre())) == "Normalized(Legendre{Float64}())"
            @test summary(Normalized(Legendre()).scaling) == "NormalizationConstant{Float64}"
        end
    end

    @testset "Chebyshev" begin
        T = ChebyshevT()
        w = ChebyshevWeight()
        wT = WeightedChebyshevT()
        Q = Normalized(T)

        @testset "Basic" begin
            @test MemoryLayout(Q) isa BasisLayout
            @test @inferred(Q\Q) ≡ Eye(∞)
        end

        @testset "recurrencecoefficients" begin
            A,B,C = recurrencecoefficients(Q)
            @test A[1] ≈ sqrt(2)
            @test A[2:5] ≈ fill(2,4)
            @test C[1:3] ≈ [0,sqrt(2),1]
            @test A[3:∞][1:10] == A[3:12]
            @test B[3:∞] ≡ Zeros(∞)
        end

        @testset "Evaluation" begin
            M = T'wT
            @test Q[0.1,1] == 1/sqrt(π)
            @test Q[0.1,2] ≈ sqrt(1/M[2,2]) * T[0.1,2]
            @test Q[0.1,Base.OneTo(10)] ≈ Q[0.1,1:10] ≈ sqrt.(inv(M)[1:10,1:10]) * T[0.1,Base.OneTo(10)]
            @test (Q'*(w .* Q))[1:10,1:10] ≈ I
        end

        @testset "Expansion" begin
            f = Q*[1:5; zeros(∞)]
            @test f[0.1] ≈ Q[0.1,1:5]'*(1:5) ≈ f[[0.1]][1]
            x = axes(f,1)
            w = Q * (Q \ (1 .- x.^2));
            @test w[0.1] ≈ (1-0.1^2) ≈ w[[0.1]][1]
        end

        @testset "Conversion" begin
            @test ((T \ Q) * (Q \ T))[1:10,1:10] ≈ (Q \Q)[1:10,1:10] ≈ I
            @test (ChebyshevU() \ Q)[1:10,1:10] ≈ ((ChebyshevU() \ T) * (T \ Q))[1:10,1:10]
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

            w = T * (T \ (1 .- x.^2));
            W = Q \ (w .* Q)
            @test W isa Clenshaw
            @test bandwidths(W) == (2,2)
            W̃ = Q\  (w .* Q)
            @test W[1:10,1:10] ≈ W[1:10,1:10]' ≈ W̃[1:10,1:10]
        end
    end

    @testset "Jacobi" begin
        P = Normalized(Jacobi(1/2,0))
        # Emperical from Mathematica
        @test P[0.1,1:4] ≈ [0.728237657560985,0.41715052371131806,-0.6523500049588019,-0.5607891513201705]
    end

    @testset "Mapped" begin
        P = legendre(0..1)
        x = axes(P,1)
        Q = Normalized(P)

        # Emperical from Mathematica
        @test Q[0.1,1:4] ≈ [1,-1.3856406460551018,1.028591269649903,-0.21166010488516684]

        u = Q[:,1:20] * (Q[:,1:20] \ exp.(x))
        @test u[0.1] ≈ exp(0.1)
        u = Q * (Q \ exp.(x))
        @test u[0.1] ≈ exp(0.1)

        Q = Normalized(jacobi(1/2,0,0..1))
        wQ = weighted(Q)
        x = axes(Q,1)
        @test wQ[0.1,1:10] ≈ Q[0.1,1:10] * sqrt(1-(2*0.1-1))
        u = wQ[:,1:20] * (wQ[:,1:20] \  @.(sqrt(1-x^2)))
        @test u[0.1] ≈ sqrt(1-0.1^2)
        u = wQ * (wQ \ @.(sqrt(1-x^2)))
        @test u[0.1] ≈ sqrt(1-0.1^2)
    end
end
