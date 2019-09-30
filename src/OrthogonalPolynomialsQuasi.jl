module OrthogonalPolynomialsQuasi
using ContinuumArrays, QuasiArrays, LazyArrays, FillArrays, BandedMatrices, IntervalSets, DomainSets, InfiniteLinearAlgebra, LinearAlgebra

import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy,
                first, last
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, adjointlayout, LdivApplyStyle
import LinearAlgebra: pinv
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import FillArrays: AbstractFill, getindex_value

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle


import ContinuumArrays: Basis, @simplify, Identity

export Jacobi, Legendre, Chebyshev, Ultraspherical,
            JacobiWeight, ChebyshevWeight, UltrasphericalWeight,
            fullmaterialize

abstract type OrthogonalPolynomial{T} <: Basis{T} end

@simplify *(B::Identity, C::OrthogonalPolynomial) = C*jacobimatrix(C)

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::OrthogonalPolynomial) 
    x == axes(C,1) || throw(DimensionMismatch())
    C*jacobimatrix(C)
end
  
function forwardrecurrence!(v::AbstractVector{T}, b::AbstractVector, a::AbstractVector, c::AbstractVector, x) where T
    isempty(v) && return v
    v[1] = one(x) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = (x-a[1])/c[1]
    @inbounds for n=3:length(v)
        v[n] = muladd(x-a[n-1],v[n-1],-b[n-1]*v[n-2])/c[n-1]
    end
    v
end

function forwardrecurrence!(v::AbstractVector{T}, b::AbstractVector, ::Zeros{<:Any,1}, c::AbstractVector, x) where T
    isempty(v) && return v
    v[1] = one(x) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = x/c[1]
    @inbounds for n=3:length(v)
        v[n] = muladd(x,v[n-1],-b[n-1]*v[n-2])/c[n-1]
    end
    v
end

# special case for Chebyshev
function forwardrecurrence!(v::AbstractVector{T}, b::AbstractVector, ::Zeros{<:Any,1}, c::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractVector}}, x) where T
    isempty(v) && return v
    c0,c∞ = c.args
    v[1] = one(x) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = x/c0
    @inbounds for n=3:length(v)
        v[n] = muladd(x,v[n-1],-b[n-2]*v[n-2])/c∞[n-2]
    end
    v
end

function forwardrecurrence!(v::AbstractVector{T}, b_v::AbstractFill, ::Zeros{<:Any,1}, c::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, x) where T
    isempty(v) && return v
    c0,c∞_v = c.args
    b = getindex_value(b_v)
    c∞ = getindex_value(c∞_v) 
    mbc  = -b/c∞
    xc = x/c∞
    v[1] = one(x) # assume OPs are normalized to one for now
    length(v) == 1 && return v
    v[2] = x/c0
    @inbounds for n=3:length(v)
        v[n] = muladd(xc,v[n-1],mbc*v[n-2])
    end
    v
end

_vec(a) = vec(a)
_vec(a::Adjoint{<:Any,<:AbstractVector}) = a'
bands(J) = _vec.(J.data.args)

function getindex(P::OrthogonalPolynomial{T}, x::Number, n::OneTo) where T
    J = jacobimatrix(P)
    b,a,c = bands(J)
    forwardrecurrence!(similar(n,T),b,a,c,x)
end

function getindex(P::OrthogonalPolynomial{T}, x::AbstractVector, n::OneTo) where T
    J = jacobimatrix(P)
    b,a,c = bands(J)
    V = Matrix{T}(undef,length(x),length(n))
    for k = eachindex(x)
        forwardrecurrence!(view(V,k,:),b,a,c,x[k])
    end
    V
end

getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][n]

getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][:,n]    

getindex(P::OrthogonalPolynomial, x::Number, n::Number) = P[x,OneTo(n)][end]


include("jacobi.jl")
include("ultraspherical.jl")



end # module
