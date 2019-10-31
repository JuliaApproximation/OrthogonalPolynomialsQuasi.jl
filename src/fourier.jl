
struct Fourier{T} <: Basis{T} end
Fourier() = Fourier{Float64}()

axes(F::Fourier) = (Inclusion(ℝ), OneTo(∞))

function getindex(F::Fourier{T}, x::Real, j::Int)::T where T
    isodd(j) && return cos((j÷2)*x)
    sin((j÷2)*x)
end

import BlockBandedMatrices: _BlockSkylineMatrix

@simplify function *(A::QuasiAdjoint{<:Any,<:Fourier}, B::Fourier)
    Diagonal(Vcat(2π,Fill(π,∞)))
end

@simplify *(D::Derivative, F::Fourier) = 
    Fourier()*_BlockBandedMatrix(Vcat(0,mortar(Fill([1,0,0,1],∞))), ([1; Fill(2,∞)], [1; Fill(2,∞)]), (0,0))

