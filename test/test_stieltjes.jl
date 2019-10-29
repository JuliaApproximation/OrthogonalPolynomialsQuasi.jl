using OrthogonalPolynomialsQuasi, ContinuumArrays, DomainSets, Test
import OrthogonalPolynomialsQuasi: Hilbert, StieltjesPoint

@testset "Chebyshev" begin
    wT = ChebyshevWeight() .* Chebyshev()
    x = axes(wT,1)
    z = 0.1+0.2im
    S = inv.(z .- x')
    @test S isa StieltjesPoint{ComplexF64,Float64,ChebyshevInterval{Float64}}
    
end

S*wT
H = inv.(x .- x')
@test H isa Hilbert{Float64,ChebyshevInterval{Float64}}
Ultraspherical(1) \ (H*wT)

x = Inclusion(0..1)
wT2 = wT[2x .- 1,:]
H = inv.(x .- x')
(H*wT2)


1 .-x
(1-x)./(1+x)

H = inv.(y .- y')
@test H isa MappedHilbert
(H*wT[y,:]).args

typeof(x)





(J' - z*I)

(S'wT)

Chebyshev()'wT

p0 = Legendre()[:,1]
p0'p0

import ApproxFun, SingularIntegralEquations
import ApproxFun: Fun
x = ApproxFun.Fun()
SingularIntegralEquations.Hilbert() : ApproxFun.JacobiWeight(-0.5,-0.5,ApproxFun.Chebyshev())
SingularIntegralEquations.Hilbert() : ApproxFun.JacobiWeight(-0.5,-0.5,ApproxFun.Chebyshev(0..1))

SingularIntegralEquations.stieltjes(Fun(ApproxFun.JacobiWeight(-0.5,-0.5,ApproxFun.Chebyshev()),[0,0,1.0]),z)

J = jacobimatrix(S.args[2])


@which factorize(J)
@time ((J-z*I) \ [1; zeros(∞)])



qr(J-z*I)


k'S


H = inv.(x .- x')

C*S


z = 
C = inv.(z .- x')




