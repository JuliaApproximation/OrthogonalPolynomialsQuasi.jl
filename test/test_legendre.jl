@testset "Legendre" begin
    @testset "basics" begin
        P = Legendre()
        @test axes(P) == (Inclusion(ChebyshevInterval()),Base.OneTo(∞))
        @test P == P == Legendre{Float32}()
    end

    @testset "operators" begin
        @test jacobimatrix(Jacobi(0.,0.))[1,1] == 0.0
        @test jacobimatrix(Jacobi(0.,0.))[1:10,1:10] == jacobimatrix(Legendre())[1:10,1:10] == jacobimatrix(Ultraspherical(1/2))[1:10,1:10]
        @test Jacobi(0.,0.)[0.1,1:10] ≈ Legendre()[0.1,1:10] ≈ Ultraspherical(1/2)[0.1,1:10]

        P = Legendre()
        P̃ = Jacobi(0.0,0.0)
        P̄ = Ultraspherical(1/2)

        @test Ultraspherical(P) == P̄
        @test Jacobi(P) == P̃

        @test P̃\P === P\P̃ === P̄\P === P\P̄ === Eye(∞)
        @test_broken P̄\P̃ === P̃\P̄ === Eye(∞)

        D = Derivative(axes(P,1))
        @test Ultraspherical(3/2)\(D*P) isa BandedMatrix{Float64,<:Fill}
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

        J = P \ (x .* P)
        @test (P \ (   (1 .+ x) .* P))[1:10,1:10] ≈ (I + J)[1:10,1:10]

        x = Inclusion(0..1)
        Q = P[2x.-1,:]
        x .* Q
    end
end