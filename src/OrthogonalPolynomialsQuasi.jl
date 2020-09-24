module OrthogonalPolynomialsQuasi
using ContinuumArrays, QuasiArrays, LazyArrays, FillArrays, BandedMatrices, BlockArrays,
    IntervalSets, DomainSets, ArrayLayouts, SpecialFunctions,
    InfiniteLinearAlgebra, InfiniteArrays, LinearAlgebra, FastGaussQuadrature, FastTransforms, FFTW

import Base: @_inline_meta, axes, getindex, convert, prod, *, /, \, +, -,
                IndexStyle, IndexLinear, ==, OneTo, tail, similar, copyto!, copy,
                first, last, Slice, size, length, axes, IdentityUnitRange, sum, _sum,
                to_indices, _maybetail, tail, getproperty, inv, show, isapprox
import Base.Broadcast: materialize, BroadcastStyle, broadcasted
import LazyArrays: MemoryLayout, Applied, ApplyStyle, flatten, _flatten, colsupport, adjointlayout,
                sub_materialize, arguments, sub_paddeddata, paddeddata, PaddedLayout, resizedata!, LazyVector, ApplyLayout, call,
                _mul_arguments, CachedVector, CachedMatrix, LazyVector, LazyMatrix, axpy!, AbstractLazyLayout, BroadcastLayout
import ArrayLayouts: MatMulVecAdd, materialize!, _fill_lmul!, sublayout, sub_materialize, lmul!, ldiv!, transposelayout, triangulardata
import LinearAlgebra: pinv, factorize, qr, adjoint, transpose
import BandedMatrices: AbstractBandedLayout, AbstractBandedMatrix, _BandedMatrix, bandeddata
import FillArrays: AbstractFill, getindex_value

import QuasiArrays: cardinality, checkindex, QuasiAdjoint, QuasiTranspose, Inclusion, SubQuasiArray,
                    QuasiDiagonal, MulQuasiArray, MulQuasiMatrix, MulQuasiVector, QuasiMatMulMat,
                    ApplyQuasiArray, ApplyQuasiMatrix, LazyQuasiArrayApplyStyle, AbstractQuasiArrayApplyStyle,
                    LazyQuasiArray, LazyQuasiVector, LazyQuasiMatrix, LazyLayout, LazyQuasiArrayStyle,
                    _getindex, layout_getindex, _factorize

import InfiniteArrays: OneToInf, InfAxes, InfUnitRange, Infinity
import ContinuumArrays: Basis, Weight, basis, @simplify, Identity, AbstractAffineQuasiVector, ProjectionFactorization,
    inbounds_getindex, grid, transform, transform_ldiv, TransformFactorization, QInfAxes, broadcastbasis, Expansion,
    AffineQuasiVector, AffineMap, WeightLayout, WeightedBasisLayout, WeightedBasisLayouts
import FastTransforms: Λ, forwardrecurrence, forwardrecurrence!, _forwardrecurrence!, clenshaw, clenshaw!,
                        _forwardrecurrence_next, _clenshaw_next, check_clenshaw_recurrences, ChebyshevGrid, chebyshevpoints

import FastGaussQuadrature: jacobimoment

import BlockArrays: blockedrange, _BlockedUnitRange, unblock, _BlockArray
import BandedMatrices: bandwidths

export OrthogonalPolynomial, Normalized, orthonormalpolynomial, LanczosPolynomial, Hermite, Jacobi, Legendre, Chebyshev, ChebyshevT, ChebyshevU, ChebyshevInterval, Ultraspherical, Fourier,
            HermiteWeight, JacobiWeight, ChebyshevWeight, ChebyshevGrid, ChebyshevTWeight, ChebyshevUWeight, UltrasphericalWeight, LegendreWeight,
            WeightedUltraspherical, WeightedChebyshev, WeightedChebyshevT, WeightedChebyshevU, WeightedJacobi,
            ∞, Derivative, .., Inclusion, chebyshevt, chebyshevu, legendre, jacobi, jacobimatrix, jacobiweight, legendreweight, chebyshevtweight, chebyshevuweight


include("interlace.jl")

# ambiguity error
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{InfAxes,QInfAxes}) = V
sub_materialize(_, V::AbstractQuasiArray, ::Tuple{QInfAxes,InfAxes}) = V

#
# BlockQuasiArrays

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{Block{1}, Vararg{Any}}) =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockRange{1,R}, Vararg{Any}}) where R =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)
@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{BlockIndex{1}, Vararg{Any}}) =
    (inds[1][I[1]], to_indices(A, _maybetail(inds), tail(I))...)

cardinality(::FullSpace{<:AbstractFloat}) = ℵ₁
cardinality(::EuclideanDomain) = ℵ₁

checkpoints(::ChebyshevInterval) = [-0.823972,0.01,0.3273484]
checkpoints(::UnitInterval) = [0.823972,0.01,0.3273484]
checkpoints(d::AbstractInterval) = width(d) .* checkpoints(UnitInterval()) .+ leftendpoint(d)
checkpoints(x::Inclusion) = checkpoints(x.domain)
checkpoints(A::AbstractQuasiMatrix) = checkpoints(axes(A,1))

transform_ldiv(A, f, ::Tuple{<:Any,Infinity})  = adaptivetransform_ldiv(A, f)

function chop!(c::AbstractVector, tol::Real)
    @assert tol >= 0

    for k=length(c):-1:1
        if abs(c[k]) > tol
            resize!(c,k)
            return c
        end
    end
    resize!(c,0)
    c
end

setaxis(c, ::OneToInf) = c
setaxis(c, ax::BlockedUnitRange) = PseudoBlockVector(c, (ax,))

function adaptivetransform_ldiv(A::AbstractQuasiArray{U}, f::AbstractQuasiArray{V}) where {U,V}
    T = promote_type(U,V)

    r = checkpoints(A)
    fr = f[r]
    maxabsfr = norm(fr,Inf)

    tol = 20eps(T)

    for n = 2 .^ (4:∞)
        An = A[:,OneTo(n)]
        cfs = An \ f
        maxabsc = maximum(abs, cfs)
        if maxabsc == 0 && maxabsfr == 0
            return zeros(T,∞)
        end

        un = A * [cfs; Zeros{T}(∞)]
        # we allow for transformed coefficients being a different size
        ##TODO: how to do scaling for unnormalized bases like Jacobi?
        if maximum(abs,@views(cfs[n-2:end])) < 10tol*maxabsc &&
                all(norm.(un[r] - fr, 1) .< tol * n * maxabsfr*1000)
            return setaxis([chop!(cfs, tol); zeros(T,∞)], axes(A,2))
        end
    end
    error("Have not converged")
end

abstract type OrthogonalPolynomial{T} <: Basis{T} end

# OPs are immutable
copy(a::OrthogonalPolynomial) = a

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

const WeightedOrthogonalPolynomial{T, A<:AbstractQuasiVector, B<:OrthogonalPolynomial} = WeightedBasis{T, A, B}

"""
    singularities(f)

gives the singularity structure of an expansion, e.g.,
`JacobiWeight`.
"""
singularities(::WeightLayout, w) = w
singularities(lay::BroadcastLayout, a) = singularitiesbroadcast(call(a), map(singularities, arguments(lay, a))...)
singularities(::WeightedBasisLayouts, a) = singularities(BroadcastLayout{typeof(*)}(), a)
singularities(w) = singularities(MemoryLayout(w), w)
singularities(f::Expansion) = singularities(basis(f))
singularities(S::WeightedOrthogonalPolynomial) = singularities(S.args[1])

singularities(S::SubQuasiArray) = singularities(parent(S))[parentindices(S)[1]]

struct NoSingularities end

singularities(::Number) = NoSingularities()
singularities(r::Base.RefValue) = r[] # pass through



orthogonalityweight(P::SubQuasiArray{<:Any,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) =
    orthogonalityweight(parent(P))[parentindices(P)[1]]

_weighted(w, P) = w .* P
weighted(P::AbstractQuasiMatrix) = _weighted(orthogonalityweight(P), P)

OrthogonalPolynomial(w::Weight) =error("Override for $(typeof(w))")

@simplify *(B::Identity, C::OrthogonalPolynomial) = C*jacobimatrix(C)

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::OrthogonalPolynomial)
    x == axes(C,1) || throw(DimensionMismatch())
    C*jacobimatrix(C)
end

# function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::BroadcastQuasiVector, C::OrthogonalPolynomial)
#     axes(a,1) == axes(C,1) || throw(DimensionMismatch())
#     # re-expand in OP basis
#     broadcast(*, C * (C \ a), C)
# end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), a::AbstractAffineQuasiVector, C::OrthogonalPolynomial)
    x = axes(C,1)
    axes(a,1) == x || throw(DimensionMismatch())
    broadcast(*, C * (C \ a), C)
end

function broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::Inclusion, C::WeightedOrthogonalPolynomial)
    x == axes(C,1) || throw(DimensionMismatch())
    w,P = C.args
    P2, J = (x .* P).args
    @assert P == P2
    (w .* P) * J
end

##
# Multiplication for mapped and subviews x .* view(P,...)
##

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

function jacobimatrix(C::SubQuasiArray{T,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) where T
    P = parent(C)
    kr,jr = parentindices(C)
    Y = jacobimatrix(P)
    kr.A \ (Y - kr.b * Eye{T}(size(Y,1)))
end

function recurrencecoefficients(C::SubQuasiArray{T,2,<:Any,<:Tuple{AbstractAffineQuasiVector,Slice}}) where T
    P = parent(C)
    kr,jr = parentindices(C)
    A,B,C = recurrencecoefficients(P)
    A * kr.A, A*kr.b + B, C
end


_vec(a) = vec(a)
_vec(a::InfiniteArrays.ReshapedArray) = _vec(parent(a))
_vec(a::Adjoint{<:Any,<:AbstractVector}) = a'
bands(J::AbstractBandedMatrix) = _vec.(bandeddata(J).args)
bands(J::Tridiagonal) = J.du, J.d, J.dl
bands(D::Diagonal{T}) where T = Zeros{T}(∞), D.diag, Zeros{T}(∞)
function bands(B::BroadcastArray{<:Any,2,<:Any,<:NTuple{2,AbstractMatrix}})
    ((au,ad,al),(bu,bd,bl)) = map(bands, B.args)
    (B.f(au,bu), B.f(ad,bd), B.f(al,bl))
end
function bands(B::BroadcastArray{<:Any,2,<:Any,<:Tuple{Number,AbstractMatrix}})
    a = B.args[1]
    (bu,bd,bl) = bands(B.args[2])
    (B.f(a,bu), B.f(a,bd), B.f(a,bl))
end

include("clenshaw.jl")


function factorize(L::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{<:Inclusion,<:OneTo}}) where T
    p = grid(L)
    TransformFactorization(p, nothing, qr(L[p,:])) # Use QR so type-stable
end

function factorize(L::SubQuasiArray{T,2,<:OrthogonalPolynomial,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T
    _,jr = parentindices(L)
    ProjectionFactorization(factorize(parent(L)[:,Base.OneTo(maximum(jr))]), jr)
end

function \(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial})
    axes(A,1) == axes(B,1) || throw(DimensionMismatch())
    _,jA = parentindices(A)
    _,jB = parentindices(B)
    (parent(A) \ parent(B))[jA, jB]
end

function \(wA::WeightedOrthogonalPolynomial, wB::WeightedOrthogonalPolynomial)
    w_A,A = arguments(wA)
    w_B,B = arguments(wB)
    w_A == w_B || error("Not implemented")
    A\B
end

include("normalized.jl")
include("lanczos.jl")
include("hermite.jl")
include("jacobi.jl")
include("chebyshev.jl")
include("ultraspherical.jl")
include("fourier.jl")
include("stieltjes.jl")


end # module
