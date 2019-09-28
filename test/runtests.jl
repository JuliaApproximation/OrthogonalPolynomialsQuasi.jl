using OrthogonalPolynomialsQuasi, ContinuumArrays, QuasiArrays, FillArrays, LazyArrays, BandedMatrices, LinearAlgebra, ForwardDiff, Test
import ContinuumArrays: SimplifyStyle
import OrthogonalPolynomialsQuasi: jacobimatrix, ∞
import LazyArrays: ApplyStyle, colsupport, MemoryLayout
import QuasiArrays: MulQuasiMatrix


@testset "Ultraspherical" begin
    T = Chebyshev()
    U = Ultraspherical(1)
    C = Ultraspherical(2)
    D = Derivative(axes(T,1))

    @test T\T === pinv(T)*T === Eye(∞)
    @test U\U === pinv(U)*U === Eye(∞)
    @test C\C === pinv(C)*C === Eye(∞)

    @test ApplyStyle(*,typeof(D),typeof(T)) == SimplifyStyle()
    @test D*T isa MulQuasiMatrix
    D₀ = U\(D*T)
    @test D₀ isa BandedMatrix
    @test D₀[1:10,1:10] isa BandedMatrix{Float64}
    @test D₀[1:10,1:10] == diagm(1 => 1:9)
    @test colsupport(D₀,1) == 1:0

    D₁ = C\(D*U)
    @test D₁ isa BandedMatrix
    @test apply(*,D₁,D₀)[1:10,1:10] == diagm(2 => 4:2:18)
    @test (D₁*D₀)[1:10,1:10] == diagm(2 => 4:2:18)
    @test D₁*D₀ isa MulMatrix
    @test bandwidths(D₁*D₀) == (-2,2)

    S₀ = (U\T)[1:10,1:10]
    @test S₀ isa BandedMatrix{Float64}
    @test S₀ == diagm(0 => [1.0; fill(0.5,9)], 2=> fill(-0.5,8))

    S₁ = (C\U)[1:10,1:10]
    @test S₁ isa BandedMatrix{Float64}
    @test S₁ == diagm(0 => 1 ./ (1:10), 2=> -(1 ./ (3:10)))
end

@testset "Legendre" begin
    @test jacobimatrix(Jacobi(0.,0.))[1,1] == 0.0
    @test jacobimatrix(Jacobi(0.,0.))[1:10,1:10] == jacobimatrix(Legendre())[1:10,1:10] == jacobimatrix(Ultraspherical(1/2))[1:10,1:10]
    @test Jacobi(0.,0.)[0.1,1:10] ≈ Legendre()[0.1,1:10] ≈ Ultraspherical(1/2)[0.1,1:10]

    P = Legendre()
    D = Derivative(axes(P,1))
    @test Ultraspherical(3/2)\(D*P) isa BandedMatrix{Float64,<:Fill}
end

@testset "Jacobi" begin
    b,a = 0.1,0.2
    S = Jacobi(b,a)
    x = 0.1
    @test S[x,1] === 1.0
    X = jacobimatrix(S)
    @test X[1,1] ≈ (a^2-b^2)/((a+b)*(a+b+2))
    @test X[2,1] ≈ 2/(a+b+2)
    @test S[x,2] ≈ 0.065
    @test S[x,10] ≈ 0.22071099583604945

    w = JacobiWeight(b,a)
    @test w[x] ≈ (1+x)^b * (1-x)^a
    wS = w.*S
    @test wS[0.1,1] ≈ w[0.1]
    @test wS[0.1,1:2] ≈ w[0.1] .* S[0.1,1:2]
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
    @test A[1:10,1:10] == diagm(1 => 1:0.5:5)

    M = @inferred(D*S)
    @test M isa MulQuasiMatrix
    @test M.args[1] == Jacobi(2,2)
    @test M.args[2][1:10,1:10] == A[1:10,1:10]
end

@testset "Weighted Jacobi integer" begin
    S = Jacobi(true,true)
    w̃ = JacobiWeight(true,false)
    A = @inferred(Jacobi(false,true)\(w̃ .* S))
    @test A isa BandedMatrix
    @test size(A) == (∞,∞)

    w̄ = JacobiWeight(false,true)
    A = @inferred(Jacobi(true,false)\(w̄.*S))
    @test A isa BandedMatrix
    @test size(A) == (∞,∞)

    w = JacobiWeight(true,true)
    P = Legendre()
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
    @test typeof(M) == typeof(A*B*C)
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
    
    F = qr(B);
    F.R
    MemoryLayout(typeof(B))


    B\ Vcat(randn(10), Zeros(∞))
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

    f = x -> Chebyshev{eltype(x)}()[x,5]
    @test ForwardDiff.derivative(f,0.1) ≈ 4*U[0.1,4]
    f = x -> Chebyshev{eltype(x)}()[x,5][1]
    @test ForwardDiff.gradient(f,[0.1]) ≈ [4*U[0.1,4]]
    @test ForwardDiff.hessian(f,[0.1]) ≈ [8*C[0.1,3]]

    f = x -> Chebyshev{eltype(x)}()[x,1:5]
    @test ForwardDiff.derivative(f,0.1) ≈ [0;(1:4).*U[0.1,1:4]]
end

@testset "∞-dimensional Dirichlet" begin
    S = Jacobi(true,true)
    W = Diagonal(JacobiWeight(true,true)) 
    D = Derivative(axes(S,1))
    X = Diagonal(Inclusion(axes(S,1)))

    (Legendre() \ S)*(S\(W*S)
    Ultraspherical(3/2)\(D^2*W*S)
end