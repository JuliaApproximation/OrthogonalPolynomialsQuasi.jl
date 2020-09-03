using OrthogonalPolynomialsQuasi, FillArrays, BandedMatrices, ContinuumArrays, QuasiArrays, LazyArrays, Test
import OrthogonalPolynomialsQuasi: recurrencecoefficients

@testset "Jacobi" begin
    @testset "basis" begin
        b,a = 0.1,0.2
        P = Jacobi(b,a)
        @test P[0.1,2] ≈ 0.16499999999999998
        P = Jacobi(a,b)
        @test P[-0.1,2] ≈ -0.16499999999999998
    end
    @testset "operators" begin
        b,a = 0.2,0.1
        S = Jacobi(b,a)
        x = 0.1
        @test S[x,1] === 1.0
        X = jacobimatrix(S)
        @test X[1,1] ≈ (b^2-a^2)/((a+b)*(a+b+2))
        @test X[2,1] ≈ 2/(a+b+2)
        @test S[x,2] ≈ 0.065
        @test S[x,10] ≈ 0.22071099583604945

        w = JacobiWeight(b,a)
        @test w[x] ≈ (1+x)^b * (1-x)^a
        wS = w.*S
        @test wS[0.1,1] ≈ w[0.1]
        @test wS[0.1,1:2] ≈ w[0.1] .* S[0.1,1:2]
    end
    @testset "functions" begin
        b,a = 0.1,0.2
        P = Jacobi(b,a)
        D = Derivative(axes(P,1))

        f = P*Vcat(randn(10), Zeros(∞))
        @test (Jacobi(b+1,a) * (Jacobi(b+1,a)\f))[0.1] ≈ f[0.1]
        h = 0.0000001
        @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=100h

        (D*(JacobiWeight(b,a) .* f))
    end

    @testset "expansions" begin
        P = Jacobi(0.,1/2)
        x = axes(P,1)
        @test (P * (P \ exp.(x)))[0.1] ≈ exp(0.1)

        wP = WeightedJacobi(0.,1/2)
        f = @.(sqrt(1 - x) * exp(x))
        @test wP[0.1,1:100]'*(wP[:,1:100] \ f) ≈ sqrt(1-0.1) * exp(0.1)
        @test (wP * (wP \ f))[0.1] ≈ sqrt(1-0.1) * exp(0.1)

        P̃ = P[affine(Inclusion(0..1), x), :]
        x̃ = axes(P̃, 1)
        @test (P̃ * (P̃ \ exp.(x̃)))[0.1] ≈ exp(0.1)
        wP̃ = wP[affine(Inclusion(0..1), x), :]
        f̃ = @.(sqrt(1 - x̃) * exp(x̃))
        @test wP̃[0.1,1:100]'*(wP̃[:,1:100] \ f̃) ≈ sqrt(1-0.1) * exp(0.1)
        @test (wP̃ * (wP̃ \ f̃))[0.1] ≈ sqrt(1-0.1) * exp(0.1)
    end

    @testset "trivial weight" begin
        S = JacobiWeight(0.0,0.0) .* Jacobi(0.0,0.0)
        @test S == S
        @test Legendre() == S
        @test Legendre()\S isa Eye
    end

    @testset "Jacobi integer" begin
        S = Jacobi(true,true)
        D = Derivative(axes(S,1))
        P = Legendre()

        @test pinv(pinv(S)) === S
        @test P\P === pinv(P)*P === Eye(∞)

        Bi = pinv(Jacobi(2,2))
        @test Bi isa QuasiArrays.PInvQuasiMatrix

        A = Jacobi(2,2) \ (D*S)
        @test typeof(A) == typeof(pinv(Jacobi(2,2))*(D*S))
        @test A isa BandedMatrix
        @test bandwidths(A) == (-1,1)
        @test size(A) == (∞,∞)
        @test A[1:10,1:10] == diagm(1 => 2:0.5:6)

        M = @inferred(D*S)
        @test M isa MulQuasiMatrix
        @test M.args[1] == Jacobi(2,2)
        @test M.args[2][1:10,1:10] == A[1:10,1:10]
    end

    @testset "Weighted Jacobi integer" begin
        S = Jacobi(true,true)
        w̃ = JacobiWeight(true,false)
        A = Jacobi(false,true)\(w̃ .* S)
        @test A isa BandedMatrix
        @test size(A) == (∞,∞)
        @test A[1:10,1:10] ≈ (Jacobi(0.0,1.0) \ (JacobiWeight(1.0,0.0) .* Jacobi(1.0,1.0)))[1:10,1:10]

        w̄ = JacobiWeight(false,true)
        A = Jacobi(true,false)\(w̄.*S)
        @test A isa BandedMatrix
        @test size(A) == (∞,∞)
        @test A[1:10,1:10] ≈ (Jacobi(1.0,0.0) \ (JacobiWeight(0.0,1.0) .* Jacobi(1.0,1.0)))[1:10,1:10]

        P = Legendre()
        w̄ = JacobiWeight(false,true)
        @test_broken P \ (w̃ .* Jacobi(false,true))
        w̄ = JacobiWeight(true,false)
        @test (P \ (w̃ .* Jacobi(true,false)))[1:10,1:10] == diagm(0 => ones(10), -1 => ones(9))

        w = JacobiWeight(true,true)
        A,B = (P'P),P\(w.*S)

        M = Mul(A,B)
        @test M[1,1] == 4/3

        M = ApplyMatrix{Float64}(*,A,B)
        M̃ = M[1:10,1:10]
        @test M̃ isa BandedMatrix
        @test bandwidths(M̃) == (2,0)

        @test A*B isa MulMatrix
        @test bandwidths(A*B) == bandwidths(B)

        A,B,C = (P\(w.*S))',(P'P),P\(w.*S)
        M = ApplyArray(*,A,B,C)
        @test bandwidths(M) == (2,2)
        @test M[1,1] ≈  1+1/15
        M = A*B*C
        @test bandwidths(M) == (2,2)
        @test M[1,1] ≈  1+1/15

        S = Jacobi(1.0,1.0)
        w = JacobiWeight(1.0,1.0)
        wS = w .* S

        W = Diagonal(w)
        @test W[0.1,0.2] ≈ 0.0
    end

    @testset "Jacobi and Chebyshev" begin
        T = ChebyshevT()
        U = ChebyshevU()
        JT = Jacobi(T)
        JU = Jacobi(U)
        
        @testset "recurrence degenerecies" begin
            A,B,C = recurrencecoefficients(JT)
            @test A[1] == 0.5
            @test B[1] == 0.0
        end

        @test JT[0.1,1:4] ≈ [1.0,0.05,-0.3675,-0.0925]

        @test ((T \ JT) * (JT \ T))[1:10,1:10] ≈ Eye(10)
        @test ((U \ JU) * (JU \ U))[1:10,1:10] ≈ Eye(10)

        @test T[0.1,1:10]' ≈ JT[0.1,1:10]' * (JT \ T)[1:10,1:10]
        @test U[0.1,1:10]' ≈ JU[0.1,1:10]' * (JU \ U)[1:10,1:10]

        w = JacobiWeight(1,1)
        @test (w .* U)[0.1,1:10] ≈ (T * (T \ (w .* U)))[0.1,1:10]
        @test (w .* U)[0.1,1:10] ≈ (JT * (JT \ (w .* U)))[0.1,1:10]
        @test (w .* JU)[0.1,1:10] ≈ (T * (T \ (w .* JU)))[0.1,1:10]
        @test (w .* JU)[0.1,1:10] ≈ (JT * (JT \ (w .* JU)))[0.1,1:10]
    end

    @testset "Jacobi-Chebyshev-Ultraspherical transforms" begin
        @test Jacobi(0.0,0.0) \ Legendre() == Eye(∞)
        @test ((Ultraspherical(3/2) \ Jacobi(1,1))*(Jacobi(1,1) \ Ultraspherical(3/2)))[1:10,1:10] ≈ Eye(10)
        f = Jacobi(0.0,0.0)*[[1,2,3]; zeros(∞)]
        g = (Legendre() \ f) - f.args[2]
        @test_skip norm(g) ≤ 1E-15
        @test_broken (Legendre() \ f) == f.args[2]
        @test (Legendre() \ f)[1:10] ≈ f.args[2][1:10]
        f = Jacobi(1.0,1.0)*[[1,2,3]; zeros(∞)]
        g = Ultraspherical(3/2)*(Ultraspherical(3/2)\f)
        @test f[0.1] ≈ g[0.1]

        @testset "Chebyshev-Legendre" begin
            T = Chebyshev()
            P = Legendre()
            @test T[:,Base.OneTo(5)] \ P[:,Base.OneTo(5)] == (T\P)[1:5,1:5]

            x = axes(P,1)
            u = P * (P \ exp.(x))
            @test u[0.1] ≈ exp(0.1)

            P = Legendre{BigFloat}()
            x = axes(P,1)
            u = P * (P \ exp.(x))
            @test u[BigFloat(1)/10] ≈ exp(BigFloat(1)/10)
        end
    end

    @testset "hcat" begin
        L = LinearSpline(range(-1,1;length=2))
        S = JacobiWeight(1.0,1.0) .* Jacobi(1.0,1.0)
        P = apply(hcat,L,S)
        @test P isa ApplyQuasiArray
        @test axes(P) == axes(S)
        V = view(P,0.1,1:10)
        @test all(arguments(V) .≈ [L[0.1,:], S[0.1,1:8]])
        @test P[0.1,1:10] == [L[0.1,:]; S[0.1,1:8]]
        D = Derivative(axes(P,1))
        # applied(*,D,P) |> typeof
        # MemoryLayout(typeof(D))
    end

    @testset "Jacobi Clenshaw" begin
        P = Jacobi(0.1,0.2)
        x = axes(P,1)
        a = P * (P \ exp.(x))
        M = P \ (a .* P);
        u = [randn(1000); zeros(∞)];
        @test (P * (M*u))[0.1] ≈ (P*u)[0.1]*exp(0.1)
    end
end