using OrthogonalPolynomialsQuasi, BandedMatrices, LazyArrays, Test

@testset "Ultraspherical" begin
    @testset "Transforms" begin
        U = Ultraspherical(1)
        x = axes(U,1)
        Un = U[:,Base.OneTo(5)]
        @test factorize(Un) isa ContinuumArrays.TransformFactorization
        @test (Un \ x) ≈ [0,0.5,0,0,0]
        @test (U * (U \ exp.(x)))[0.1] ≈ exp(0.1)
    end

    @testset "Operators" begin
        @testset "Lowering" begin
            λ = 1
            wC1 = WeightedUltraspherical(λ)
            wC2 = WeightedUltraspherical(λ+1)
            L = wC1 \ wC2
            @test L isa BandedMatrix
            @test bandwidths(L) == (2,0)
            @test L[1,1] ≈ 2λ*(2λ+1)/(4λ*(λ+1))
            @test L[3,1] ≈ -2/(4λ*(λ+1))
            u = [randn(10); zeros(∞)]
            @test wC1[0.1,:]'*(L*u) ≈ wC2[0.1,:]'*u
        end

        @testset "Weighted Derivative" begin
            T = Chebyshev()
            wC1 = WeightedUltraspherical(1)
            wC2 = WeightedUltraspherical(2)
            x = axes(wC2,1)
            D = Derivative(x)

            @test wC1 \ (D*wC2) isa BandedMatrix
            @test (wC1 \ (D*wC2))[1:5,1:5] == BandedMatrix(-1 => [-1.5,-4,-7.5,-12])

            u = wC2 * [randn(5); zeros(∞)]
            @test (D*u)[0.1] ≈ ((D*T) * (T\u))[0.1]
        end

        @testset "Interrelationships" begin
            @testset "Chebyshev–Ultrashperical" begin
                T = Chebyshev()
                U = ChebyshevU()
                C = Ultraspherical(2)
                D = Derivative(axes(T,1))

                @test C\C === pinv(C)*C === Eye(∞)
                D₀ = U\(D*T)
                D₁ = C\(D*U)
                @test D₁ isa BandedMatrix
                @test (D₁*D₀)[1:10,1:10] == diagm(2 => 4:2:18)
                @test D₁*D₀ isa MulMatrix
                @test bandwidths(D₁*D₀) == (-2,2)

                S₁ = (C\U)[1:10,1:10]
                @test S₁ isa BandedMatrix{Float64}
                @test S₁ == diagm(0 => 1 ./ (1:10), 2=> -(1 ./ (3:10)))
            end
            @testset "Legendre" begin
                @test Ultraspherical(0.5) \ (UltrasphericalWeight(0.0) .* Ultraspherical(0.5)) == Eye(∞)
                @test Legendre() \ (UltrasphericalWeight(0.0) .* Ultraspherical(0.5)) == Eye(∞)
            end
        end
    end

    @testset "test on functions" begin
        T = Chebyshev()
        U = Ultraspherical(1)
        D = Derivative(axes(T,1))
        f = T*Vcat(randn(10), Zeros(∞))
        @test (U*(U\f)).args[1] isa Ultraspherical
        @test (U*(U\f))[0.1] ≈ f[0.1]
        @test (D*f)[0.1] ≈ ForwardDiff.derivative(x -> (ChebyshevT{eltype(x)}()*f.args[2])[x],0.1)
    end

    @testset "Evaluation" begin
        C = Ultraspherical(2)
        @test @inferred(C[0.1,Base.OneTo(0)]) == Float64[]
        @test @inferred(C[0.1,Base.OneTo(1)]) == [1.0]
        @test @inferred(C[0.1,Base.OneTo(2)]) == [1.0,0.4]
        @test @inferred(C[0.1,Base.OneTo(3)]) == [1.0,0.4,-1.88]
    end

end