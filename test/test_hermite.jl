using OrthogonalPolynomialsQuasi, ContinuumArrays, DomainSets, Test
import OrthogonalPolynomialsQuasi: jacobimatrix

@testset "Hermite" begin
    H = Hermite()
    @test axes(H) == (Inclusion(ℝ), Base.OneTo(∞))
    x = axes(H,1)
    X = jacobimatrix(H)
    X*X

    @test H[0.1,1] === 1.0 # equivalent to H_0(0.1) == 1.0
    D = Derivative(x)
    
    h = 0.000001
    @test (D*H)[0.1,1:5] ≈ (H[0.1+h,1:5] - H[0.1,1:5])/h atol=100h
end