
mutable struct NormalizationConstant{T, DL, DU} <: LazyVector{T}
    dl::DL # subdiagonal of Jacobi
    du::DU # superdiagonal
    data::Vector{T}
    datasize::Tuple{Int}

    NormalizationConstant{T, DL, DU}(μ::T, dl::DL, du::DU) where {T,DL,DU} = new{T, DL, DU}(dl, du, [μ], (1,))
end

NormalizationConstant(μ::T, dl::AbstractVector{T}, du::AbstractVector{T}) where T = NormalizationConstant{T,typeof(dl),typeof(du)}(μ, dl, du)

function NormalizationConstant(P::OrthogonalPolynomial)
    dl, _, du = bands(jacobimatrix(P))
    NormalizationConstant(inv(sqrt(sum(weight(P)))), dl, du)
end

size(K::NormalizationConstant) = (∞,)
# Behaves like a CachedVector
getindex(K::NormalizationConstant, k) = LazyArrays.cache_getindex(K, k)
getindex(K::NormalizationConstant, k::AbstractVector) = LazyArrays.cache_getindex(K, k)
getindex(K::NormalizationConstant, k::InfUnitRange) = layout_getindex(K, k)
getindex(K::SubArray{<:Any,1,<:NormalizationConstant}, k::InfUnitRange) = layout_getindex(K, k)

resizedata!(B::NormalizationConstant, mn...) = resizedata!(MemoryLayout(typeof(B.data)), UnknownLayout(), B, mn...)
function LazyArrays.cache_filldata!(K::NormalizationConstant, inds)
    @inbounds for k in inds
        K.data[k] = sqrt(K.du[k-1]/K.dl[k]) * K.data[k-1]
    end
end

struct Normalized{T, OPs<:OrthogonalPolynomial{T}, NL} <: OrthogonalPolynomial{T}
    P::OPs
    scaling::NL # Q = P * Diagonal(scaling)
end

Normalized(P::OrthogonalPolynomial{T}) where T = Normalized(P, NormalizationConstant(P))
axes(Q::Normalized) = axes(Q.P)

_p0(Q::Normalized) = Q.scaling[1]

function recurrencecoefficients(Q::Normalized)
    A,B,C = recurrencecoefficients(Q.P)
    h = Q.scaling
    h[2:∞] ./ h .* A, h[2:∞] ./ h .* B, Vcat(zero(eltype(Q)), h[3:∞] ./ h .* C[2:∞])
end

MemoryLayout(::Type{<:Normalized}) = ApplyLayout{typeof(*)}()
arguments(::ApplyLayout{typeof(*)}, Q::Normalized) = Q.P, Diagonal(Q.scaling)


# p_{n+1} = (A_n * x + B_n) * p_n - C_n * p_{n-1}
# q_{n+1}/h[n+1] = (A_n * x + B_n) * q_n/h[n] - C_n * p_{n-1}/h[n-1]
# q_{n+1} = (h[n+1]/h[n] * A_n * x + h[n+1]/h[n] * B_n) * q_n - h[n+1]/h[n-1] * C_n * p_{n-1}





# function symmetrize_jacobi(J)
#     d=Array{T}(undef, n)
#     d[1]=1
#     for k=2:n
#         d[k]=sqrt(J[k,k-1]/J[k-1,k])*d[k-1]
#     end

#    SymTridiagonal(
#     T[J[k,k] for k=1:n],
#     T[J[k,k+1]*d[k+1]/d[k] for k=1:n-1])
# end


