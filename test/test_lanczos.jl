using OrthogonalPolynomialsQuasi, BandedMatrices, ArrayLayouts, Test
import OrthogonalPolynomialsQuasi: recurrencecoefficients, PaddedLayout

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
        P = Normalized(Legendre())
        x = axes(P,1)

        Q = LanczosPolynomial(exp.(x))
        R = P \ Q
        @test MemoryLayout(R[:,2]) isa PaddedLayout

        A,B,C = recurrencecoefficients(Q)
        @test A[1] ≈ 1.903680130866564 # emperical from Mathematica
        @test B[1] ≈ -0.5959190532652192
        @test A[2] ≈ 1.9150612001588696
        @test B[2] ≈ 0.0845629033308663
        @test C[2] ≈ 1.005978456731134

        @test Q[0.1,1] ≈ (Q * [1; zeros(∞)])[0.1] ≈ P[0.1,:]'*R[:,1] ≈ 0.6522722316024658
        @test Q[0.1,2] ≈ (Q * [0; 1; zeros(∞)])[0.1] ≈ P[0.1,:]'*R[:,2] ≈ -0.26452968200597243
        @test Q[0.1,3] ≈ (Q * [zeros(2); 1; zeros(∞)])[0.1] ≈ P[0.1,:]'*R[:,3] ≈ -0.7292002638736375
        @test Q[0.1,5] ≈ 0.7576999562707534 # emperical

        Q = LanczosPolynomial(  1 ./ (2 .+ x));
        R = P \ Q
        @test norm(R[1,3:10]) ≤ 1E-14

        Q = LanczosPolynomial(  1 ./ (2 .+ x).^2);
        R = P \ Q
        @test norm(R[1,4:10]) ≤ 1E-14

        # polys
        Q = LanczosPolynomial( 2 .+ x);
        R = P \ Q;
        @test norm(inv(R)[1,3:10]) ≤ 1E-14

        w = P * (P \ (1 .+ x))
        Q = LanczosPolynomial(w)
        @test Q[0.1,5] ≈ Normalized(Jacobi(0,1))[0.1,5] ≈ 0.742799258138176

        Q = LanczosPolynomial( 1 .+ x.^2);
        R = P \ Q;
        @test norm(inv(R)[1,4:10]) ≤ 1E-14
    end

    @testset "Expansion" begin
        P = Legendre();
        x = axes(P,1)
        w = P * [1; zeros(∞)];
        Q = LanczosPolynomial(w);
        R = Normalized(P) \ Q
        @test R * [1; 2; 3; zeros(∞)] ≈ [R[1:3,1:3] * [1,2,3]; zeros(∞)]
        @test R \ [1; 2; 3; zeros(∞)] ≈ [1; 2; 3; zeros(∞)]
        @test (Q * (Q \ (1 .- x.^2)))[0.1] ≈ (1-0.1^2)
    end

    @testset "Jacobi via Lanczos" begin
        P = Legendre(); x = axes(P,1)
        w = P * (P \ (1 .- x.^2))
        Q = LanczosPolynomial(w)
        A,B,C = recurrencecoefficients(Q)

        @test @inferred(Q[0.1,1]) ≈ sqrt(3)/sqrt(4)
        @test Q[0.1,2] ≈ 2*0.1 * sqrt(15)/sqrt(16)
    end

    @testset "Singularity" begin
        T = Chebyshev(); wT = WeightedChebyshev()
        x = axes(T,1)
        
        w = wT * [1; zeros(∞)]
        Q = LanczosPolynomial(w)
        @test Q[0.1,1:10] ≈ Normalized(T)[0.1,1:10]
    end

    @testset "BigFloat" begin
        P = Legendre{BigFloat}()
        x = axes(P,1)
        w = P * (P \ exp.(x))
        W = P \ (w .* P)
        v = [[1,2,3]; zeros(BigFloat,∞)];
        Q = LanczosPolynomial(w)

        x̃ = BigFloat(1)/10
        @test Q[x̃,1] ≈ 0.652272231602465791008015756161075576539994569266308567422055126278763683344388252
        @test Q[x̃,2] ≈ -0.26452968200597244253463861322599173806126678155361307561211048667577270734771616
        @test Q[x̃,3] ≈ -0.72920026387366053084159259908849371062183891778315602761397748592062615496583854

        X = Q \ (x .* Q)
        # empirical test
        @test X[5,5] ≈ -0.001489975039238321407179828331585356464766466154894764141171294038822525312179884

        @test (Q*[1; 2; 3; zeros(BigFloat,∞)])[0.1] ≈ -2.0643879240304606865860392675563890314480557471903856666440983048346601962485597
        @test 0.1*(Q*[1; 2; 3; zeros(BigFloat,∞)])[0.1] ≈ (Q * (X * [1; 2; 3; zeros(BigFloat,∞)]))[0.1]
    end

    @testset "Mixed Jacobi" begin
        P = Jacobi(1/2,0)
        x = axes(P,1)
        
        w = @. sqrt(1-x)
        Q = LanczosPolynomial(w, P)
        @test Q[0.1,1:10] ≈ Normalized(P)[0.1,1:10]

        w = @. exp(x) * sqrt(1-x)
        Q = LanczosPolynomial(w, P)
        # emperical from Julia
        @test Q[0.1,10] ≈ 0.5947384257847858
    end

    @testset "Mapped" begin
        x = Inclusion(0..1)
        w = @. sqrt(1 - x^2)
        Q = LanczosPolynomial(w, jacobi(1/2,0,0..1))
        # emperical from Julia
        @test Q[0.1,10] ≈ -0.936819626414421
    end
end
