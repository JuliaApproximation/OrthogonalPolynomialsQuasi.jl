using Base, OrthogonalPolynomialsQuasi, ContinuumArrays, QuasiArrays, FillArrays, 
        LazyArrays, BandedMatrices, LinearAlgebra, FastTransforms, ForwardDiff, IntervalSets, Test
import ContinuumArrays: SimplifyStyle, BasisLayout, MappedBasisLayout
import OrthogonalPolynomialsQuasi: jacobimatrix, ∞
import LazyArrays: ApplyStyle, colsupport, MemoryLayout, arguments
import QuasiArrays: MulQuasiMatrix
import Base: OneTo

@testset "ChebyshevGrid" begin
    for kind in (1,2)
        @test all(ChebyshevGrid{kind}(10) .=== chebyshevpoints(Float64,10; kind=kind))
        for T in (Float16, Float32, Float64)
            @test all(ChebyshevGrid{kind,T}(10) .=== chebyshevpoints(T,10; kind=kind))
        end
        @test ChebyshevGrid{kind,BigFloat}(10) == chebyshevpoints(BigFloat,10; kind=kind)
    end
end

@testset "Transforms" begin
    @testset "Chebyshev" begin
        T = Chebyshev()
        Tn = @inferred(T[:,OneTo(100)])
        @test grid(Tn) == chebyshevpoints(100; kind=1)
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
        @test grid(Tn) == chebyshevpoints(100; kind=1)
        @test (Tn \ (exp.(x) .- 1.26606587775201)) ≈ (Tn \ u) ≈ (T\u)[2:100]

        u = T * (T \ exp.(x))        
        @test u[0.1] ≈ exp(0.1)

        v = T[:,2:end] \ (exp.(x) .- 1.26606587775201)
        @test v[1:10] ≈ (T\u)[2:11]
    end

    @testset "Mapped Chebyshev" begin
        x = Inclusion(0..1)
        T = Chebyshev()[2x .- 1,:]
        @test (T*(T\x))[0.1] ≈ 0.1
        @test (T* (T \ exp.(x)))[0.1] ≈ exp(0.1)
    end

    @testset "Ultraspherical" begin
        U = Ultraspherical(1)
        x = axes(U,1)
        Un = U[:,Base.OneTo(5)]
        @test factorize(Un) isa ContinuumArrays.TransformFactorization
        @test (Un \ x) ≈ [0,0.5,0,0,0]
        @test (U * (U \ exp.(x)))[0.1] ≈ exp(0.1)
    end
end

@testset "Chebyshev" begin
    @testset "operators" begin
        T = ChebyshevT()
        U = ChebyshevU()
        D = Derivative(axes(T,1))

        @test T\T === pinv(T)*T === Eye(∞)
        @test U\U === pinv(U)*U === Eye(∞)       
        
        @test ApplyStyle(*,typeof(D),typeof(T)) == SimplifyStyle()
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

    @testset "Jacobi and Chebyshev" begin
        T = ChebyshevT()
        U = ChebyshevU()
        JT = Jacobi(T)
        JU = Jacobi(U)
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
end

@testset "Ultraspherical" begin
    @testset "operators" begin
        T = Chebyshev()
        U = ChebyshevU()
        C = Ultraspherical(2)
        D = Derivative(axes(T,1))

        @test C\C === pinv(C)*C === Eye(∞)
        D₀ = U\(D*T)
        D₁ = C\(D*U)
        @test D₁ isa BandedMatrix
        @test apply(*,D₁,D₀)[1:10,1:10] == diagm(2 => 4:2:18)
        @test (D₁*D₀)[1:10,1:10] == diagm(2 => 4:2:18)
        @test D₁*D₀ isa MulMatrix
        @test bandwidths(D₁*D₀) == (-2,2)

        S₁ = (C\U)[1:10,1:10]
        @test S₁ isa BandedMatrix{Float64}
        @test S₁ == diagm(0 => 1 ./ (1:10), 2=> -(1 ./ (3:10)))
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
end


@testset "Legendre" begin
    @testset "operators" begin
        @test jacobimatrix(Jacobi(0.,0.))[1,1] == 0.0
        @test jacobimatrix(Jacobi(0.,0.))[1:10,1:10] == jacobimatrix(Legendre())[1:10,1:10] == jacobimatrix(Ultraspherical(1/2))[1:10,1:10]
        @test Jacobi(0.,0.)[0.1,1:10] ≈ Legendre()[0.1,1:10] ≈ Ultraspherical(1/2)[0.1,1:10]

        P = Legendre()
        P̃ = Jacobi(0.0,0.0)
        P̄ = Ultraspherical(1/2)
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
end

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

    @testset "trivial weight" begin
        S = JacobiWeight(0.0,0.0) .* Jacobi(0.0,0.0)
        @test S == S
        @test Legendre() == S
        @test Legendre()\S isa Eye
    end
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

    @test A*B isa BroadcastArray
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

@testset "P-FEM" begin
    S = Jacobi(true,true)
    w = JacobiWeight(true,true)
    D = Derivative(axes(S,1))
    P = Legendre()

    @test w.*S isa QuasiArrays.BroadcastQuasiMatrix
    
    M = P\(D*(w.*S))
    @test M isa BandedMatrix
    @test M[1:10,1:10] == diagm(-1 => -2.0:-2:-18.0)

    N = 10
    A = D* (w.*S)[:,1:N]
    @test A.args[1] == P    
    @test P\(D*(w.*S)[:,1:N]) isa MulMatrix

    L = D*(w.*S)
    Δ = L'L
    @test Δ isa MulMatrix
    @test Δ[1:3,1:3] isa BandedMatrix
    @test bandwidths(Δ) == (0,0)

    L = D*(w.*S)[:,1:N]

    A  = apply(*, (L').args..., L.args...)
    @test A isa MulQuasiMatrix

    A  = *((L').args..., L.args...)
    @test A isa MulQuasiMatrix

    @test apply(*,L',L) isa QuasiArrays.ApplyQuasiArray
    Δ = L'L
    @test Δ isa MulMatrix
    @test bandwidths(Δ) == (0,0)
end

@testset "∞-FEM" begin
    S = Jacobi(true,true)
    w = JacobiWeight(true,true)
    D = Derivative(axes(w,1))
    WS = w.*S
    L = D* WS
    Δ = L'L
    P = Legendre()
    
    f = P * Vcat(randn(10), Zeros(∞))
    (P\WS)'*(P'P)*(P\WS)
    B = BroadcastArray(+, Δ, (P\WS)'*(P'P)*(P\WS))
    @test colsupport(B,1) == 1:3
    
    @test axes(B.args[2].args[1]) == (Base.OneTo(∞),Base.OneTo(∞))
    @test axes(B.args[2]) == (Base.OneTo(∞),Base.OneTo(∞))
    @test axes(B) == (Base.OneTo(∞),Base.OneTo(∞))

    @test BandedMatrix(view(B,1:10,13:20)) == zeros(10,8)

    F = qr(B);
    b = Vcat(randn(10), Zeros(∞))
    @test B*(F \ b) ≈ b
end

@testset "Chebyshev evaluation" begin
    P = Chebyshev()
    @test @inferred(P[0.1,Base.OneTo(0)]) == Float64[]
    @test @inferred(P[0.1,Base.OneTo(1)]) == [1.0]
    @test @inferred(P[0.1,Base.OneTo(2)]) == [1.0,0.1]
    for N = 1:10
        @test @inferred(P[0.1,Base.OneTo(N)]) ≈ @inferred(P[0.1,1:N]) ≈ [cos(n*acos(0.1)) for n = 0:N-1]
        @test @inferred(P[0.1,N]) ≈ cos((N-1)*acos(0.1))
    end
    @test P[0.1,[2,5,10]] ≈ [0.1,cos(4acos(0.1)),cos(9acos(0.1))]

    P = Ultraspherical(1)
    @test @inferred(P[0.1,Base.OneTo(0)]) == Float64[]
    @test @inferred(P[0.1,Base.OneTo(1)]) == [1.0]
    @test @inferred(P[0.1,Base.OneTo(2)]) == [1.0,0.2]
    for N = 1:10
        @test @inferred(P[0.1,Base.OneTo(N)]) ≈ @inferred(P[0.1,1:N]) ≈ [sin((n+1)*acos(0.1))/sin(acos(0.1)) for n = 0:N-1]
        @test @inferred(P[0.1,N]) ≈ sin(N*acos(0.1))/sin(acos(0.1))
    end
    @test P[0.1,[2,5,10]] ≈ [0.2,sin(5acos(0.1))/sin(acos(0.1)),sin(10acos(0.1))/sin(acos(0.1))]

    P = Ultraspherical(2)
    @test @inferred(P[0.1,Base.OneTo(0)]) == Float64[]
    @test @inferred(P[0.1,Base.OneTo(1)]) == [1.0]
    @test @inferred(P[0.1,Base.OneTo(2)]) == [1.0,0.4]
    @test @inferred(P[0.1,Base.OneTo(3)]) == [1.0,0.4,-1.88]
end

@testset "Collocation" begin
    P = Chebyshev()
    D = Derivative(axes(P,1))
    n = 300
    x = cos.((0:n-2) .* π ./ (n-2))
    cfs = [P[-1,1:n]'; (D*P)[x,1:n] - P[x,1:n]] \ [exp(-1); zeros(n-1)]
    u = P[:,1:n]*cfs
    @test u[0.1] ≈ exp(0.1)

    P = Chebyshev()
    D = Derivative(axes(P,1))
    D2 = D*(D*P) # could be D^2*P in the future
    n = 300
    x = cos.((1:n-2) .* π ./ (n-1)) # interior Chebyshev points 
    C = [P[-1,1:n]';
         D2[x,1:n] + P[x,1:n];
         P[1,1:n]']
    cfs = C \ [1; zeros(n-2); 2] # Chebyshev coefficients
    u = P[:,1:n]*cfs  # interpret in basis
    @test u[0.1] ≈ (3cos(0.1)sec(1) + csc(1)sin(0.1))/2
end

@testset "Auto-diff" begin
    U = Ultraspherical(1)
    C = Ultraspherical(2)

    f = x -> ChebyshevT{eltype(x)}()[x,5]
    @test ForwardDiff.derivative(f,0.1) ≈ 4*U[0.1,4]
    f = x -> ChebyshevT{eltype(x)}()[x,5][1]
    @test ForwardDiff.gradient(f,[0.1]) ≈ [4*U[0.1,4]]
    @test ForwardDiff.hessian(f,[0.1]) ≈ [8*C[0.1,3]]

    f = x -> ChebyshevT{eltype(x)}()[x,1:5]
    @test ForwardDiff.derivative(f,0.1) ≈ [0;(1:4).*U[0.1,1:4]]
end

@testset "∞-dimensional Dirichlet" begin
    S = Jacobi(true,true)
    w = JacobiWeight(true,true)
    D = Derivative(axes(S,1))
    X = Diagonal(Inclusion(axes(S,1)))

    @test_broken (Legendre() \ S)*(S\(w.*S))
    @test (Ultraspherical(3/2)\(D^2*(w.*S)))[1:10,1:10] ≈ diagm(0 => -(2:2:20))
end

@testset "rescaled" begin 
    x = Inclusion(0..1)
    S = Jacobi(1.0,1.0)[2x.-1,:]
    D = Derivative(x)
    f = S*[[1,2,3]; zeros(∞)]
    g = Jacobi(1.0,1.0)*[[1,2,3]; zeros(∞)]
    @test f[0.1] ≈ g[2*0.1-1]
    h = 0.0000001
    @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=100h
    @test Jacobi(2.0,2.0)[2x.-1,:] \ (D*S).args[1] isa BandedMatrix
    @test (Jacobi(2.0,2.0)[2x.-1,:] \ (D*S))[1:10,1:10] == diagm(1 => 4:12)

    P = Legendre()[2x.-1,:]
    w = JacobiWeight(1.0,1.0)
    wS = (w .* Jacobi(1.0,1.0))[2x.-1,:]
    @test MemoryLayout(typeof(wS)) isa MappedBasisLayout
    f = wS*[[1,2,3]; zeros(∞)]
    g = (w .* Jacobi(1.0,1.0))*[[1,2,3]; zeros(∞)]
    @test f[0.1] ≈ g[2*0.1-1]
    h = 0.0000001
    @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=100h
    @test P == P

    @test P.parent == (D*wS).args[1].parent
    DwS = apply(*,D,wS)
    A,B = P,arguments(DwS)[1];
    @test (A.parent\B.parent) == Eye(∞)
    @test (P \ (DwS))[1:10,1:10] == diagm(-1 => -4:-4:-36)
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
end

@testset "Beam" begin
    P = JacobiWeight(0.0,0.0) .* Jacobi(0.0,0.0)
    x = axes(P,1)
    D = Derivative(x)
    @test (D*P).args[1] == Jacobi{Float64}(1,1)
    @test (Jacobi(1,1)\(D*P))[1:10,1:10] ≈ (Jacobi(1,1) \ (D*Legendre()))[1:10,1:10]

    S = JacobiWeight(2.0,2.0) .* Jacobi(2.0,2.0)
    @test (Legendre() \ S)[1,1] ≈ 0.533333333333333333
    Δ² = (D^2*S)'*(D^2*S)
    M = S'S
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