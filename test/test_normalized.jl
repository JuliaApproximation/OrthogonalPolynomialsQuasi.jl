using OrthogonalPolynomialsQuasi
import OrthogonalPolynomialsQuasi: NormalizationConstant, recurrencecoefficients, Normalized


@testset "Normalized" begin
    P = Legendre()
    Q = Normalized(P)
    M = P'P

    @test sqrt(2) * Q[0.1,Base.OneTo(10)] ≈ sqrt(2) * Q[0.1,1:10] ≈ sqrt.(M[1:10,1:10])* P[0.1,Base.OneTo(10)]
    @test sqrt(2) * Q[0.1,1:10] ≈ sqrt.(M[1:10,1:10])* P[0.1,Base.OneTo(10)]
    @test (Q'Q)[1:10,1:10] ≈ Matrix(2I, 10, 10)

    D = Derivative(axes(Q,1))
    f = Q*[1:5; zeros(∞)]
    h = 0.000001
    @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=1E-4
end
