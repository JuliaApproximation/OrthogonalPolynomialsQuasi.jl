using OrthogonalPolynomialsQuasi, BandedMatrices, Test
import OrthogonalPolynomialsQuasi: recurrencecoefficients

@testset "Lanczos" begin
    @testset "Legendre" begin
        P = Legendre();
        w = P * [1; zeros(∞)];
        Q = LanczosPolynomial(w);
        @test Q.data.W[1:10,1:10] isa BandedMatrix

        Q̃ = Normalized(P);
        A,B,C = recurrencecoefficients(Q);
        Ã,B̃,C̃ = recurrencecoefficients(Q̃);
        @test @inferred(A[1:10]) ≈ Ã[1:10] ≈ [A[k] for k=1:10]
        @test @inferred(B[1:10]) ≈ B̃[1:10] ≈ [B[k] for k=1:10]
        @test @inferred(C[2:10]) ≈ C̃[2:10] ≈ [C[k] for k=2:10]

        @test A[1:10] isa Vector{Float64}
        @test B[1:10] isa Vector{Float64}
        @test C[1:10] isa Vector{Float64}

        @test Q[0.1,1:10] ≈ Q̃[0.1,1:10]

        R = P \ Q
        @test R[1:10,1:10] ≈ (P \ Q̃)[1:10,1:10]
        @test (Q'Q)[1:10,1:10] ≈ I


        # Q'Q == I => Q*sqrt(M) = P

        x = axes(P,1)
        X = Q' * (x .* Q)
        X̃ = Q̃' * (x .* Q̃)
        @test X[1:10,1:10] ≈ X̃[1:10,1:10]
    end

    @testset "other weight" begin
        P = Legendre()
        x = axes(P,1)

        Q = LanczosPolynomial(exp.(x))
        @test Q[0.1,5] ≈ 0.48479905558644537 # emperical

        Q = LanczosPolynomial(  1 ./ (2 .+ x));
        R = P \ Q
        @test norm(R[1,3:10]) ≤ 1E-14

        Q = LanczosPolynomial(  1 ./ (2 .+ x).^2);
        R = P \ Q
        @test norm(R[1,4:10]) ≤ 1E-14

        # polys
        Q = LanczosPolynomial( 2 .+ x);
        R = P \ Q;
        Ri = inv(R)

        w = P * (P \ (1 .+ x))
        Q = LanczosPolynomial(w)
        P \ Q

        Jacobi(1,0) \ Legendre()

        Q = LanczosPolynomial((x -> 1+x).(x));
        R = P \ Q;

        Q = LanczosPolynomial( 1 .+ x.^2);
        R = P \ Q;
        @test norm(inv(R[1:10,1:10])[1,4:10]) ≤ 1E-14
    end

    @testset "Expansion" begin
        P = Legendre();
        w = P * [1; zeros(∞)];
        Q = LanczosPolynomial(w);
        R = Normalized(P) \ Q
        @test R * [1; 2; 3; zeros(∞)] ≈ [R[1:3,1:3] * [1,2,3]; zeros(∞)]
        @test R \ [1; 2; 3; zeros(∞)] ≈ [1; 2; 3; zeros(∞)]
        @test (Q * (Q \ (1 .- x.^2)))[0.1] ≈ (1-0.1^2)
        Q \ P
    end

    

    @testset "Jacobi via Lanczos" begin
        P = Legendre(); x = axes(P,1)
        w = P * (P \ (1 .- x.^2))
        Q = LanczosPolynomial(w)
        A,B,C = recurrencecoefficients(Q)

        @test @inferred(Q[0.1,1]) ≈ sqrt(3)/sqrt(4)
        @test Q[0.1,2] ≈ 2*0.1 * sqrt(15)/sqrt(16)
    end

    @testset "broadcast" begin
        x = Inclusion(ChebyshevInterval())
        Q = LanczosPolynomial(exp.(x))
        # Emperical
        @test Q[0.1,2] ≈ 0.4312732517146977
    end

    @testset "Singularity" begin
        T = Chebyshev(); wT = WeightedChebyshev()
        x = axes(T,1)
        @testset "Recover ChebyshevT" begin
            w = wT * [1; zeros(∞)]
            Q = LanczosPolynomial(w)
            @test Q[0.1,1:10] ≈ Normalized(T)[0.1,1:10]
        end
    end

    @testset "BigFloat" begin
        P = Legendre{BigFloat}()
        x = axes(P,1)
        w = P * (P \ exp.(x))
        W = P \ (w .* P)
        v = [[1,2,3]; zeros(BigFloat,∞)];
        Q = LanczosPolynomial(w)
        X = Q \ (x .* Q)
        # empirical test
        @test X[5,5] ≈ -0.001489975039238321407179828331585356464766466154894764141171294038822525312179884
    end
end
