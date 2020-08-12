
mutable struct NormalizationConstant{T, DL, DU} <: LazyVector{T}
    dl::DL # subdiagonal of Jacobi
    du::DU # superdiagonal
    data::Vector{T}
    datasize::Tuple{Int}

    NormalizationConstant{T, DL, DU}(dl::DL, du::DU) where {T,DL,DU} = new{T, DL, DU}(dl, du, [one(T)], (1,))
end

NormalizationConstant(dl::AbstractVector{T}, du::AbstractVector{T}) where T = NormalizationConstant{T,typeof(dl),typeof(du)}(dl, du)

function NormalizationConstant(P::OrthogonalPolynomial)
    dl, _, du = bands(jacobimatrix(P))
    NormalizationConstant(dl, du)
end

size(K::NormalizationConstant) = (∞,)
# Behaves like a CachedVector
getindex(K::NormalizationConstant, k) = LazyArrays.cache_getindex(K, k)

resizedata!(B::NormalizationConstant, mn...) = resizedata!(MemoryLayout(typeof(B.data)), UnknownLayout(), B, mn...)
function LazyArrays.cache_filldata!(K::NormalizationConstant, inds)
    @inbounds for k in inds
        K.data[k] = K.dl[k]/K.du[k-1] * K.data[k-1]
    end
end

massmatrix(P) = Diagonal(NormalizationConstant(P))



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


# struct Normalized{T, OPs<:OrthogonalPolynomial{T}} <: OrthogonalPolynomial{T}
#     P::OPs
# end

# Normalized(P::OrthogonalPolynomial{T}) where T = Normalized{T, typeof(P)}(P)
# axes(Q::Normalized) = axes(Q.P)
