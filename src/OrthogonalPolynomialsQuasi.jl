module OrthogonalPolynomialsQuasi
using ContinuumArrays, QuasiArrays, LazyArrays, FillArrays, BandedMatrices, BlockArrays,
    IntervalSets, DomainSets,
    InfiniteLinearAlgebra, InfiniteArrays, LinearAlgebra, FastTransforms

import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy,
                first, last, Slice, size, length, axes, IdentityUnitRange, sum, _sum,
                to_indices, _maybetail, tail
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, adjointlayout, LdivApplyStyle, sub_materialize
import LinearAlgebra: pinv, factorize
import BandedMatrices: AbstractBandedLayout, _BandedMatrix
import FillArrays: AbstractFill, getindex_value

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle,
                    _getindex, layout_getindex, _factorize

import InfiniteArrays: OneToInf, InfAxes
import ContinuumArrays: Basis, Weight, @simplify, Identity, AbstractAffineQuasiVector, ProjectionFactorization,
    inbounds_getindex, grid, transform, transform_ldiv, TransformFactorization, QInfAxes
import FastTransforms: Λ

export Hermite, Jacobi, Legendre, Chebyshev, ChebyshevT, ChebyshevU, Ultraspherical, Fourier,
            HermiteWeight, JacobiWeight, ChebyshevWeight, ChebyshevGrid, ChebyshevTWeight, ChebyshevUWeight, UltrasphericalWeight,
            WeightedUltraspherical, WeightedChebyshev, WeightedJacobi,
            ∞, Derivative

import BlockArrays: blockedrange, _BlockedUnitRange, unblock, _BlockArray

# ambiguity error
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{InfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,InfAxes}) = V

#
# BlockQuasiArrays

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{Block{1}, Vararg{Any}}) =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockRange{1,R}, Vararg{Any}}) where R =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)

cardinality(::FullSpace{<:AbstractFloat}) = ℵ₁
cardinality(::EuclideanDomain) = ℵ₁

checkpoints(::ChebyshevInterval) = [-0.823972,0.01,0.3273484]
checkpoints(::UnitInterval) = [0.823972,0.01,0.3273484]
checkpoints(d::AbstractInterval) = width(d) .* checkpoints(UnitInterval()) .+ leftendpoint(d)
checkpoints(x::Inclusion) = checkpoints(x.domain)
checkpoints(A::AbstractQuasiMatrix) = checkpoints(axes(A,1))

transform_ldiv(A, f, ::Tuple{<:Any,OneToInf})  = adaptivetransform_ldiv(A, f)
transform_ldiv(A, f, ::Tuple{<:Any,IdentityUnitRange{<:OneToInf}})  = adaptivetransform_ldiv(A, f)
transform_ldiv(A, f, ::Tuple{<:Any,Slice{<:OneToInf}})  = adaptivetransform_ldiv(A, f)

function     adaptivetransform_ldiv(A::AbstractQuasiArray{U}, f::AbstractQuasiArray{V}) where {U,V}
    T = promote_type(U,V)

    r = checkpoints(A)
    fr = f[r]
    maxabsfr = norm(fr,Inf)

    tol = eps(T)

    for n = 2 .^ (4:∞)
        An = A[:,OneTo(n)]
        cfs = An \ f
        maxabsc = maximum(abs, cfs)
        if maxabsc == 0 && maxabsfr == 0
            return zeros(T,∞)
        end

        un = ApplyQuasiArray(*, An, cfs)
        # we allow for transformed coefficients being a different size
        ##TODO: how to do scaling for unnormalized bases like Jacobi?
        if maximum(abs,@views(cfs[n-8:end])) < 10tol*maxabsc &&
                all(norm.(un[r] - fr, 1) .< tol * n * maxabsfr*1000)
            return [cfs; zeros(T,∞)]
        end
    end
    error("Have not converged")
end

abstract type OrthogonalPolynomial{T} <: Basis{T} end


"""
    jacobimatrix(S)

returns the Jacobi matrix `X` associated to a quasi-matrix of orthogonal polynomials
satisfying
```julia
x = axes(S,1)
x*S == S*X
```
Note that `X` is the transpose of the usual definition of the Jacobi matrix.
"""
jacobimatrix(S) = error("Override for $(typeof(S))")

@simplify *(B::Identity, C::OrthogonalPolynomial) = C*jacobimatrix(C)

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::OrthogonalPolynomial)
    x == axes(C,1) || throw(DimensionMismatch())
    C*jacobimatrix(C)
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), y::AbstractAffineQuasiVector, C::OrthogonalPolynomial)
    x = axes(C,1)
    axes(y,1) == x || throw(DimensionMismatch())
    broadcast(+, y.A * (x.*C), y.b.*C)
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::WeightedBasis{<:Any,<:Any,<:OrthogonalPolynomial})
    x == axes(C,1) || throw(DimensionMismatch())
    w,P = C.args
    P2, J = (x .* P).args
    @assert P == P2
    (w .* P) * J
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::SubQuasiArray{<:Any,2,<:Any,<:Tuple{<:AbstractAffineQuasiVector,<:Any}})
    T = promote_type(eltype(x), eltype(C))
    x == axes(C,1) || throw(DimensionMismatch())
    P = parent(C)
    kr,jr = parentindices(C)
    y = axes(P,1)
    Y = P \ (y .* P)
    X = kr.A \ (Y     - kr.b * Eye{T}(∞))
    P[kr, :] * view(X,:,jr)
end

function forwardrecurrence!(v::AbstractVector, b::AbstractVector, a::AbstractVector, c::AbstractVector, x, shift=0)
    isempty(v) && return v
    p0 = one(x) # assume OPs are normalized to one for now
    p1 = (x-a[1])/c[1]
    @inbounds for n = 1:shift
        p1,p0 = muladd(x-a[n-1],v[n-1],-b[n-1]*v[n-2])/c[n-1],p1
    end
    v[1] = p0
    length(v) == 1 && return v
    v[2] = p1
    @inbounds for n = 3:length(v)
        p1,p0 = muladd(x-a[n-1],v[n-1],-b[n-1]*v[n-2])/c[n-1],p1
        v[n] = p1
    end
    v
end

function forwardrecurrence!(v::AbstractVector, b::AbstractVector, ::Zeros{<:Any,1}, c::AbstractVector, x, shift=0)
    isempty(v) && return v
    p0 = one(x) # assume OPs are normalized to one for now
    p1 = x/c[1]
    @inbounds for n = 1:shift
        p1,p0 = muladd(x,p1,-b[n-1]*p0)/c[n-1],p1
    end
    v[1] = p0
    length(v) == 1 && return v
    v[2] = p1
    @inbounds for n = 3:length(v)
        p1,p0 = muladd(x,p1,-b[n-1]*p0)/c[n-1],p1
        v[n] = p1
    end
    v
end

# special case for Chebyshev
function forwardrecurrence!(v::AbstractVector, b_v::AbstractFill, ::Zeros{<:Any,1}, c::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, x, shift=0)
    isempty(v) && return v
    c0,c∞_v = c.args
    b = getindex_value(b_v)
    c∞ = getindex_value(c∞_v)
    mbc  = -b/c∞
    xc = x/c∞
    p0 = one(x) # assume OPs are normalized to one for now
    p1 = x/c0
    for n = 1:shift
        p1,p0 = muladd(xc,p1,mbc*p0),p1
    end
    v[1] = p0
    length(v) == 1 && return v
    v[2] = p1
    @inbounds for n = 3:length(v)
        p1,p0 = muladd(xc,p1,mbc*p0),p1
        v[n] = p1
    end
    v
end

_vec(a) = vec(a)
_vec(a::InfiniteArrays.ReshapedArray) = _vec(parent(a))
_vec(a::Adjoint{<:Any,<:AbstractVector}) = a'
bands(J) = _vec.(J.data.args)

function getindex(P::OrthogonalPolynomial{T}, x::Number, n::OneTo) where T
    J = jacobimatrix(P)
    b,a,c = bands(J)
    forwardrecurrence!(similar(n,T),b,a,c,x)
end

getindex(P::OrthogonalPolynomial{T}, x::AbstractVector, n::AbstractUnitRange{Int}) where T =
    copyto!(Matrix{T}(undef,length(x),length(n)), view(P, x, n))


function copyto!(dest::AbstractArray, V::SubArray{<:Any,2,<:OrthogonalPolynomial,<:Tuple{<:AbstractVector,<:UnitRange}})
    checkbounds(dest, axes(V)...)
    P = parent(V)
    xr,jr = parentindices(V)
    J = jacobimatrix(P)
    b,a,c = bands(J)
    shift = first(jr)-1
    for (k,x) = enumerate(xr)
        forwardrecurrence!(view(dest,k,:), b, a, c, x, shift)
    end
    dest
end

function copyto!(dest::AbstractArray, V::SubArray{<:Any,1,<:OrthogonalPolynomial,<:Tuple{<:Number,<:UnitRange}})
    checkbounds(dest, axes(V)...)
    P = parent(V)
    x,jr = parentindices(V)
    J = jacobimatrix(P)
    b,a,c = bands(J)
    shift = first(jr)-1
    forwardrecurrence!(dest, b, a, c, x, shift)
    dest
end

getindex(P::OrthogonalPolynomial, x::Number, n::UnitRange) = layout_getindex(P, x, n)
getindex(P::OrthogonalPolynomial, x::AbstractVector, n::UnitRange) = layout_getindex(P, x, n)

getindex(P::OrthogonalPolynomial, x::Number, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][n]

getindex(P::OrthogonalPolynomial, x::AbstractVector, n::AbstractVector{<:Integer}) =
    P[x,OneTo(maximum(n))][:,n]

getindex(P::OrthogonalPolynomial, x::Number, n::Number) = P[x,OneTo(n)][end]


function factorize(L::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{<:Inclusion,<:OneTo}}) where T
    p = grid(L)
    TransformFactorization(p, nothing, factorize(L[p,:]))
end

function factorize(L::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T
    _,jr = parentindices(L)
    ProjectionFactorization(factorize(parent(L)[:,Base.OneTo(maximum(jr))]), jr)
end

include("hermite.jl")
include("jacobi.jl")
include("chebyshev.jl")
include("ultraspherical.jl")
include("fourier.jl")
include("stieltjes.jl")


end # module
