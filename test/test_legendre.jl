using OrthogonalPolynomialsQuasi, LazyArrays, QuasiArrays, Test
import OrthogonalPolynomialsQuasi: recurrencecoefficients, jacobimatrix
import QuasiArrays: MulQuasiArray

@testset "Legendre" begin
    @testset "basics" begin
        P = Legendre()
        @test axes(P) == (Inclusion(ChebyshevInterval()),Base.OneTo(∞))
        @test P == P == Legendre{Float32}()
        A,B,C = recurrencecoefficients(P)
        @test B isa Zeros
        P = Jacobi(0.0,0.0)
        A,B,C = recurrencecoefficients(P)
        @test B[1] == 0.0
    end

    @testset "operators" begin
        P = Legendre()
        P̃ = Jacobi(0.0,0.0)
        P̄ = Ultraspherical(1/2)

        @test jacobimatrix(P̃)[1,1] == 0.0
        @test jacobimatrix(P̃)[1:10,1:10] == jacobimatrix(P)[1:10,1:10] == jacobimatrix(P̄)[1:10,1:10]

        @test P[0.1,Base.OneTo(4)] ≈ [1,0.1,-0.485,-0.1475]
        @test P̃[0.1,1:10] ≈ P[0.1,1:10] ≈ P̄[0.1,1:10]

        @test Ultraspherical(P) == P̄
        @test Jacobi(P) == P̃

        @test P̃\P === P\P̃ === P̄\P === P\P̄ === Eye(∞)
        @test_broken P̄\P̃ === P̃\P̄ === Eye(∞)

        D = Derivative(axes(P,1))
        @test Ultraspherical(3/2)\(D*P) isa BandedMatrix{Float64,<:Ones}
    end

    @testset "test on functions" begin
        P = Legendre()
        D = Derivative(axes(P,1))
        f = P*Vcat(randn(10), Zeros(∞))
        P̃ = Jacobi(0.0,0.0)
        P̄ = Ultraspherical(1/2)
        @test (P̃*(P̃\f))[0.1] ≈ (P̄*(P̄\f))[0.1] ≈ f[0.1]
        C = Ultraspherical(3/2)
        @test (C*(C\f))[0.1] ≈ f[0.1]

        @test (D*f)[0.1] ≈ ForwardDiff.derivative(x -> (Legendre{eltype(x)}()*f.args[2])[x],0.1)
    end

    @testset "poly broadcast" begin
        P = Legendre()
        x = axes(P,1)

        @test (x .* P) isa MulQuasiArray

        J = P \ (x .* P)
        @test (P \ (   (1 .+ x) .* P))[1:10,1:10] ≈ (I + J)[1:10,1:10]

        x = Inclusion(0..1)
        Q = P[2x.-1,:]
        @test x .* Q isa MulQuasiArray
    end

    @testset "sum" begin
        P = Legendre()
        x = axes(P,1)
        w = P * (P \ exp.(x))
        @test sum(w) ≈ ℯ - inv(ℯ)
    end

    @testset "Normalized" begin
        P = Legendre()
        qr(P)
    end
end