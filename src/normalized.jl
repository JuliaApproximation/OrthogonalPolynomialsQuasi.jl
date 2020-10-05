
mutable struct NormalizationConstant{T, DL, DU} <: AbstractCachedVector{T}
    dl::DL # subdiagonal of Jacobi
    du::DU # superdiagonal
    data::Vector{T}
    datasize::Tuple{Int}

    NormalizationConstant{T, DL, DU}(μ::T, dl::DL, du::DU) where {T,DL,DU} = new{T, DL, DU}(dl, du, [μ], (1,))
end

NormalizationConstant(μ::T, dl::AbstractVector{T}, du::AbstractVector{T}) where T = NormalizationConstant{T,typeof(dl),typeof(du)}(μ, dl, du)

function NormalizationConstant(P::AbstractQuasiMatrix)
    dl, _, du = bands(jacobimatrix(P))
    NormalizationConstant(inv(sqrt(sum(orthogonalityweight(P)))), dl, du)
end

size(K::NormalizationConstant) = (∞,)

# How we populate the data
function LazyArrays.cache_filldata!(K::NormalizationConstant, inds)
    @inbounds for k in inds
        K.data[k] = sqrt(K.du[k-1]/K.dl[k]) * K.data[k-1]
    end
end


struct Normalized{T, OPs<:AbstractQuasiMatrix{T}, NL} <: OrthogonalPolynomial{T}
    P::OPs
    scaling::NL # Q = P * Diagonal(scaling)
end

normalizationconstant(P) = NormalizationConstant(P)
Normalized(P::AbstractQuasiMatrix{T}) where T = Normalized(P, normalizationconstant(P))
Normalized(Q::Normalized) = Q


struct NormalizedBasisLayout{LAY<:AbstractBasisLayout} <: AbstractBasisLayout end

MemoryLayout(::Type{<:Normalized{<:Any, OPs}}) where OPs = NormalizedBasisLayout{typeof(MemoryLayout(OPs))}()

struct QuasiQR{T, QQ, RR} <: Factorization{T}
    Q::QQ
    R::RR
end

QuasiQR(Q::AbstractQuasiMatrix{T}, R::AbstractMatrix{V}) where {T,V} =
    QuasiQR{promote_type(T,V),typeof(Q),typeof(R)}(Q, R)

Base.iterate(S::QuasiQR) = (S.Q, Val(:R))
Base.iterate(S::QuasiQR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(S::QuasiQR, ::Val{:done}) = nothing


axes(Q::Normalized) = axes(Q.P)
==(A::Normalized, B::Normalized) = A.P == B.P

# There is no point in a Normalized OP thats ==, so just return false
==(A::Normalized, B::OrthogonalPolynomial) = false
==(A::OrthogonalPolynomial, B::Normalized) = false
==(A::Normalized, B::AbstractQuasiMatrix) = false
==(A::AbstractQuasiMatrix, B::Normalized) = false
==(A::Normalized, B::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}) = false
==(A::SubQuasiArray{<:Any,2,<:OrthogonalPolynomial}, B::Normalized) = false

_p0(Q::Normalized) = Q.scaling[1]

# p_{n+1} = (A_n * x + B_n) * p_n - C_n * p_{n-1}
# q_{n+1}/h[n+1] = (A_n * x + B_n) * q_n/h[n] - C_n * p_{n-1}/h[n-1]
# q_{n+1} = (h[n+1]/h[n] * A_n * x + h[n+1]/h[n] * B_n) * q_n - h[n+1]/h[n-1] * C_n * p_{n-1}

function recurrencecoefficients(Q::Normalized)
    A,B,C = recurrencecoefficients(Q.P)
    h = Q.scaling
    h[2:∞] ./ h .* A, h[2:∞] ./ h .* B, Vcat(zero(eltype(Q)), h[3:∞] ./ h .* C[2:∞])
end

# x * p[n] = c[n-1] * p[n-1] + a[n] * p[n] + b[n] * p[n+1]
# x * q[n]/h[n] = c[n-1] * q[n-1]/h[n-1] + a[n] * q[n]/h[n] + b[n] * q[n+1]/h[n+1]
# x * q[n+1] = c[n-1] * h[n]/h[n-1] * q[n-1] + a[n] * q[n] + b[n] * h[n]/h[n+1] * q[n+1]

# q_{n+1}/h[n+1] = (A_n * x + B_n) * q_n/h[n] - C_n * p_{n-1}/h[n-1]
# q_{n+1} = (h[n+1]/h[n] * A_n * x + h[n+1]/h[n] * B_n) * q_n - h[n+1]/h[n-1] * C_n * p_{n-1}
function jacobimatrix(Q::Normalized)
    _,a,b = bands(jacobimatrix(Q.P))
    h = Q.scaling
    Symmetric(_BandedMatrix(Vcat(a', (b .* h ./ h[2:end])'), ∞, 1, 0), :L)
end

orthogonalityweight(Q::Normalized) = orthogonalityweight(Q.P)
singularities(Q::Normalized) = singularities(Q.P)

function demap(Q::Normalized)
    P,D =  arguments(ApplyLayout{typeof(*)}(), Q)
    demap(P) * D
end

# Sometimes we want to expand out, sometimes we don't

QuasiArrays.ApplyQuasiArray(Q::Normalized) = ApplyQuasiArray(*, arguments(ApplyLayout{typeof(*)}(), Q)...)

ArrayLayouts.mul(Q::Normalized, C::AbstractArray) = ApplyQuasiArray(*, Q, C)

grid(Q::SubQuasiArray{<:Any,2,<:Normalized}) = grid(view(parent(Q).P, parentindices(Q)...))

# transform_ldiv(Q::Normalized, C::AbstractQuasiArray) = Q.scaling .\ (Q.P \ C)
function transform_ldiv(Q::Normalized, C::AbstractQuasiArray)
    c = paddeddata(Q.P \ C)
    [Q.scaling[axes(c,1)] .\ c; zeros(eltype(c), ∞)]
end

function transform_ldiv(V::SubQuasiArray{<:Any,2,<:Normalized}, C::AbstractQuasiArray)
    Q = parent(V)
    P = Q.P
    kr, jr = parentindices(V)
    c = transform_ldiv(view(P, kr, jr), C)
    Q.scaling[axes(c,1)] .\ c
end

arguments(::ApplyLayout{typeof(*)}, Q::Normalized) = Q.P, Diagonal(Q.scaling)
_mul_arguments(Q::Normalized) = arguments(ApplyLayout{typeof(*)}(), Q)
_mul_arguments(Q::QuasiAdjoint{<:Any,<:Normalized}) = arguments(ApplyLayout{typeof(*)}(), Q)

# table stable identity if A.P == B.P
@inline _normalized_ldiv(An, C, Bn) = An \ (C * Bn)
@inline _normalized_ldiv(An, C::Eye{T}, Bn) where T = FillArrays.SquareEye{promote_type(eltype(An),T,eltype(Bn))}(∞)
copy(L::Ldiv{<:NormalizedBasisLayout,<:NormalizedBasisLayout}) = _normalized_ldiv(Diagonal(L.A.scaling), L.A.P \ L.B.P, Diagonal(L.B.scaling))
copy(L::Ldiv{Lay,<:NormalizedBasisLayout}) where Lay = copy(Ldiv{Lay,ApplyLayout{typeof(*)}}(L.A, L.B))
copy(L::Ldiv{<:NormalizedBasisLayout,Lay}) where Lay = copy(Ldiv{ApplyLayout{typeof(*)},Lay}(L.A, L.B))

###
# show
###
Base.array_summary(io::IO, C::NormalizationConstant{T}, inds) where T = print(io, "NormalizationConstant{$T}")
show(io::IO, Q::Normalized) = print(io, "Normalized($(Q.P))")
show(io::IO, ::MIME"text/plain", Q::Normalized) = show(io, Q)