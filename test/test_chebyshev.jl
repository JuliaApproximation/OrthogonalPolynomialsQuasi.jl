using OrthogonalPolynomialsQuasi, QuasiArrays, ContinuumArrays, BandedMatrices, LazyArrays, FastTransforms, ArrayLayouts, Test
import OrthogonalPolynomialsQuasi: Clenshaw, recurrencecoefficients, clenshaw, paddeddata, jacobimatrix
import LazyArrays: ApplyStyle
import QuasiArrays: MulQuasiMatrix
import Base: OneTo

@testset "Chebyshev" begin
    @testset "ChebyshevGrid" begin
        for kind in (1,2)
            @test all(ChebyshevGrid{kind}(10) .=== chebyshevpoints(Float64,10, Val(kind)))
            for T in (Float16, Float32, Float64)
                @test all(ChebyshevGrid{kind,T}(10) .=== chebyshevpoints(T,10, Val(kind)))
            end
            @test ChebyshevGrid{kind,BigFloat}(10) == chebyshevpoints(BigFloat,10, Val(kind))
        end
    end

    @testset "Evaluation" begin
        @testset "T" begin
            T = Chebyshev()
            @test @inferred(T[0.1,Base.OneTo(0)]) == Float64[]
            @test @inferred(T[0.1,Base.OneTo(1)]) == [1.0]
            @test @inferred(T[0.1,Base.OneTo(2)]) == [1.0,0.1]
            for N = 1:10
                @test @inferred(T[0.1,Base.OneTo(N)]) ≈ @inferred(T[0.1,1:N]) ≈ cos.((0:N-1)*acos(0.1))
                @test @inferred(T[0.1,N]) ≈ cos((N-1)*acos(0.1))
            end
            @test T[0.1,[2,5,10]] ≈ [0.1,cos(4acos(0.1)),cos(9acos(0.1))]

            @test axes(T[1:1,:]) === (Base.OneTo(1), Base.OneTo(∞))
            @test T[1:1,:][:,1:5] == ones(1,5)
            @test T[0.1,:][1:10] ≈ T[0.1,1:10] ≈ (T')[1:10,0.1]
        end

        @testset "U" begin
            U = ChebyshevU()
            @test @inferred(U[0.1,Base.OneTo(0)]) == Float64[]
            @test @inferred(U[0.1,Base.OneTo(1)]) == [1.0]
            @test @inferred(U[0.1,Base.OneTo(2)]) == [1.0,0.2]
            for N = 1:10
                @test @inferred(U[0.1,Base.OneTo(N)]) ≈ @inferred(U[0.1,1:N]) ≈ [sin((n+1)*acos(0.1))/sin(acos(0.1)) for n = 0:N-1]
                @test @inferred(U[0.1,N]) ≈ sin(N*acos(0.1))/sin(acos(0.1))
            end
            @test U[0.1,[2,5,10]] ≈ [0.2,sin(5acos(0.1))/sin(acos(0.1)),sin(10acos(0.1))/sin(acos(0.1))]
        end

        @testset "Function eval" begin
            T = Chebyshev()
            f = T * [1; 2; 3; zeros(∞)]
            @test f[0.1] ≈ 1 + 2*0.1 + 3*cos(2acos(0.1))
            x = [0.1,0.2]
            @test f[x] ≈ @.(1 + 2*x + 3*cos(2acos(x)))
        end
    end

    @testset "Transform" begin
        @testset "ChebyshevT" begin
            T = Chebyshev()
            @test T == ChebyshevT() == chebyshevt()
            Tn = @inferred(T[:,OneTo(100)])
            @test grid(Tn) == chebyshevpoints(100, Val(1))
            P = factorize(Tn)
            u = T*[P.plan * exp.(P.grid); zeros(∞)]
            @test u[0.1] ≈ exp(0.1)

            # auto-transform
            x = axes(T,1)
            u = Tn * (P \ exp.(x))
            @test u[0.1] ≈ exp(0.1)

            u = Tn * (Tn \ exp.(x))
            @test u[0.1] ≈ exp(0.1)

            Tn = T[:,2:100]
            @test factorize(Tn) isa ContinuumArrays.ProjectionFactorization
            @test grid(Tn) == chebyshevpoints(100, Val(1))
            @test (Tn \ (exp.(x) .- 1.26606587775201)) ≈ (Tn \ u) ≈ (T\u)[2:100]

            u = T * (T \ exp.(x))
            @test u[0.1] ≈ exp(0.1)

            v = T[:,2:end] \ (exp.(x) .- 1.26606587775201)
            @test v[1:10] ≈ (T\u)[2:11]
        end
        @testset "ChebyshevU" begin
            U = ChebyshevU()
            @test U == chebyshevu()
            @test U ≠ ChebyshevT()
            x = axes(U,1)
            F = factorize(U[:,Base.OneTo(5)])
            @test @inferred(F \ x) ≈ [0,0.5,0,0,0]
            v = (x -> (3/20 + x + (2/5) * x^2)*exp(x)).(x)
            @inferred(U[:,Base.OneTo(5)]\v)
            w = @inferred(U\v)
            @test U[0.1,:]'w ≈ v[0.1]
        end
    end

    @testset "Mapped Chebyshev" begin
        x = Inclusion(0..1)
        T = Chebyshev()[2x .- 1,:]
        @test (T*(T\x))[0.1] ≈ 0.1
        @test (T* (T \ exp.(x)))[0.1] ≈ exp(0.1)
        @test chebyshevt(0..1) == T

        U = chebyshevu(0..1)
        @test (U*(U\x))[0.1] ≈ 0.1
        @test (U* (U \ exp.(x)))[0.1] ≈ exp(0.1)
    end

    @testset "weighted" begin
        T = ChebyshevT()
        wT = WeightedChebyshevT()
        x = axes(wT,1)
        @test (x .* wT).args[2] isa BandedMatrix

        a = wT * [1; 2; 3; zeros(∞)];
    end

    @testset "operators" begin
        T = ChebyshevT()
        U = ChebyshevU()
        @test axes(T) == axes(U) == (Inclusion(ChebyshevInterval()),Base.OneTo(∞))
        D = Derivative(axes(T,1))

        @test T\T === pinv(T)*T === Eye(∞)
        @test U\U === pinv(U)*U === Eye(∞)

        @test D*T isa MulQuasiMatrix
        D₀ = U\(D*T)
        @test D₀ isa BandedMatrix
        @test D₀[1:10,1:10] isa BandedMatrix{Float64}
        @test D₀[1:10,1:10] == diagm(1 => 1:9)
        @test colsupport(D₀,1) == 1:0

        S₀ = (U\T)[1:10,1:10]
        @test S₀ isa BandedMatrix{Float64}
        @test S₀ == diagm(0 => [1.0; fill(0.5,9)], 2=> fill(-0.5,8))

        x = axes(T,1)
        J = T\(x.*T)
        @test J isa BandedMatrix
        @test J[1:10,1:10] == jacobimatrix(T)[1:10,1:10]
    end

    @testset "test on functions" begin
        T = ChebyshevT()
        U = ChebyshevU()
        D = Derivative(axes(T,1))
        f = T*Vcat(randn(10), Zeros(∞))
        @test (U*(U\f)).args[1] isa Chebyshev{2}
        @test (U*(U\f))[0.1] ≈ f[0.1]
        @test (D*f)[0.1] ≈ ForwardDiff.derivative(x -> (ChebyshevT{eltype(x)}()*f.args[2])[x],0.1)
    end

    @testset "U->T lowering"  begin
        wT = ChebyshevWeight() .* Chebyshev()
        wU = ChebyshevUWeight() .*  ChebyshevU()
        @test (wT \ wU)[1:10,1:10] == diagm(0 => fill(0.5,10), -2 => fill(-0.5,8))
    end

    @testset "sub-of-sub" begin
        T = Chebyshev()
        V = T[:,2:end]
        @test view(V,0.1:0.1:1,:) isa SubArray
        @test V[0.1:0.1:1,:] isa SubArray
        @test V[0.1:0.1:1,:][:,1:5] == T[0.1:0.1:1,2:6]
        @test parentindices(V[:,OneTo(5)])[1] isa Inclusion
    end

    @testset "==" begin
        @test Chebyshev() == ChebyshevT() == ChebyshevT{Float32}()
        @test ChebyshevU() == ChebyshevU{Float32}()
        @test Chebyshev{3}() == Chebyshev{3,Float32}()
        @test Chebyshev() ≠ ChebyshevU()
    end

    @testset "sum" begin
        wT = ChebyshevWeight() .* Chebyshev()
        @test sum(wT; dims=1)[1,1:10] == [π; zeros(9)]
        @test sum(wT * [[1,2,3]; zeros(∞)]) == 1.0π
        wU = ChebyshevUWeight() .* ChebyshevU()
        @test sum(wU; dims=1)[1,1:10] == [π/2; zeros(9)]
        @test sum(wU * [[1,2,3]; zeros(∞)]) == π/2

        x = Inclusion(0..1)
        @test sum(wT[2x .- 1, :]; dims=1)[1,1:10] == [π/2; zeros(9)]
        @test sum(wT[2x .- 1, :] * [[1,2,3]; zeros(∞)]) == π/2
    end

    @testset "algebra" begin
        T = ChebyshevT()
        U = ChebyshevU()

        a = T * [1; zeros(∞)]
        b = T * [1; 2; 3; zeros(∞)]
        c = U * [4; 5; zeros(∞)]

        @test a + b == b + a == T * [2;  2;  3; zeros(∞)]
        @test a - b == T * [0; -2; -3; zeros(∞)]
        @test a + c == c + a == U * [5; 5; zeros(∞)]
    end

    @testset "multiplication" begin
        U = ChebyshevU()
        x = axes(U,1)
        a = U * [[0.25; 0.5; 0.1]; Zeros(∞)];
        M = Clenshaw(a, U)

        @test colsupport(M,4:10) == rowsupport(M,4:10) == 2:12
        @test colsupport(M',4:10) == rowsupport(M',4:10) == 2:12

        A,B,C = recurrencecoefficients(U)
        c = paddeddata(a.args[2])
        m = 10
        X = (U \ (x .* U))[1:m,1:m]
        @test clenshaw(c, A, B, C, X) ≈ 3/20*I + X + (2/5) * X^2

        @test bandwidths(M) == (2,2)
        @test M[2:10,2:10] == M[1:10,1:10][2:10,2:10]
        @test M[3:10,5:20] == M[1:10,1:20][3:10,5:20]
        @test M[100_000:101_000,100_000:101_000] == M[2:1002,2:1002]

        @test Eye(∞) * M isa Clenshaw

        M = U \ (a .* U)
        @test M isa Clenshaw

        f = U \ exp.(x)
        @test M*f ≈ U \ (x -> (3/20 + x + (2/5) * x^2)*exp(x)).(x)

        @testset "weighted" begin
            T = ChebyshevT()
            wT = WeightedChebyshevT()
            a = wT * [1; 2; 3; zeros(∞)];
            @test (a .* T)[0.1,1:10] ≈ a[0.1] * T[0.1,1:10]
        end
    end
end