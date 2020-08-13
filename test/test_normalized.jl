using OrthogonalPolynomialsQuasi, FillArrays
import OrthogonalPolynomialsQuasi: NormalizationConstant, recurrencecoefficients, Normalized


@testset "Normalized" begin
    P = Legendre()
    Q = Normalized(P)

    @testset "recurrencecoefficients" begin
        A,B,C = recurrencecoefficients(Q)
        @test A[3:∞][1:10] == A[3:12]
        @test B[3:∞] ≡ Zeros(∞)
    end

    M = P'P
    @test Q[0.1,1] == 1
    @test Q[0.1,2] ≈ sqrt(2/M[2,2]) * P[0.1,2]
    @test inv(sqrt(2)) * Q[0.1,Base.OneTo(10)] ≈ inv(sqrt(2)) * Q[0.1,1:10] ≈ sqrt.(inv(M)[1:10,1:10]) * P[0.1,Base.OneTo(10)]
    @test (Q'Q)[1:10,1:10] ≈ Matrix(2I, 10, 10)

    D = Derivative(axes(Q,1))
    f = Q*[1:5; zeros(∞)]
    h = 0.000001
    @test (D*f)[0.1] ≈ (f[0.1+h]-f[0.1])/h atol=1E-4
end
