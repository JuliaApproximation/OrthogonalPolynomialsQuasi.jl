
struct Fourier{T} <: Basis{T} end
Fourier() = Fourier{Float64}()

==(::Fourier, ::Fourier) = true

axes(F::Fourier) = (Inclusion(ℝ), _BlockedUnitRange(1:2:∞))

function getindex(F::Fourier{T}, x::Real, j::Int)::T where T
    isodd(j) && return cos((j÷2)*x)
    sin((j÷2)*x)
end

import BlockBandedMatrices: _BlockSkylineMatrix

@simplify function *(A::QuasiAdjoint{<:Any,<:Fourier}, B::Fourier)
    TV = promote_type(eltype(A),eltype(B))
    Diagonal(Vcat(2convert(TV,π),Fill(convert(TV,π),∞)))
end

@simplify function *(D::Derivative, F::Fourier)
    TV = promote_type(eltype(D),eltype(F))
    Fourier{TV}()*_BlockArray(Diagonal(Vcat([reshape([0.0],1,1)], (1.0:∞) .* Fill([0 -one(TV); one(TV) 0], ∞))), (axes(F,2),axes(F,2)))
end


function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), c::BroadcastQuasiVector{<:Any,typeof(cos),<:Tuple{<:Inclusion{<:Any,<:FullSpace}}}, F::Fourier)
    axes(c,1) == axes(F,1) || throw(DimensionMismatch())
    T = promote_type(eltype(c), eltype(F))
    F*mortar(Tridiagonal(Vcat([reshape([0; one(T)],2,1)], Fill(Matrix(one(T)/2*I,2,2),∞)),
                        Vcat([zeros(T,1,1)], Fill(Matrix(zero(T)I,2,2),∞)),
                        Vcat([[0 one(T)/2]], Fill(Matrix(one(T)/2*I,2,2),∞))))
end


function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), s::BroadcastQuasiVector{<:Any,typeof(sin),<:Tuple{<:Inclusion{<:Any,<:FullSpace}}}, F::Fourier)
    axes(s,1) == axes(F,1) || throw(DimensionMismatch())
    T = promote_type(eltype(s), eltype(F))
    F*mortar(Tridiagonal(Vcat([reshape([one(T); 0],2,1)], Fill([0 one(T)/2; -one(T)/2 0],∞)),
                        Vcat([zeros(T,1,1)], Fill(Matrix(zero(T)*I,2,2),∞)),
                        Vcat([[one(T)/2 0]], Fill([0 -one(T)/2; one(T)/2 0],∞))))
end
