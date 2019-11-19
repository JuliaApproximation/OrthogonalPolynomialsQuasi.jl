const StieltjesPoint{T,V,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{BroadcastQuasiMatrix{T,typeof(-),Tuple{T,QuasiAdjoint{V,Inclusion{V,D}}}}}}
const Hilbert{T,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{BroadcastQuasiMatrix{T,typeof(-),Tuple{Inclusion{T,D},QuasiAdjoint{T,Inclusion{T,D}}}}}}
const MappedHilbert{T,D} = BroadcastQuasiMatrix{T,typeof(inv),Tuple{BroadcastQuasiMatrix{T,typeof(-),Tuple{AffineQuasiVector{T,T,Inclusion{T,D},T},QuasiAdjoint{T,AffineQuasiVector{T,T,Inclusion{T,D},T}}}}}}


@simplify function *(S::StieltjesPoint{<:Any,<:Any,<:ChebyshevInterval}, wT::WeightedBasis{<:Any,<:ChebyshevTWeight,<:ChebyshevT})
    w,T = wT.args
    J = jacobimatrix(T)
    z, x = parent(S).args[1].args
    transpose((J'-z*I) \ [-π; zeros(∞)])
end

@simplify function *(H::Hilbert{<:Any,<:ChebyshevInterval}, wT::WeightedBasis{<:Any,<:ChebyshevTWeight,<:ChebyshevT}) 
    T = promote_type(eltype(H), eltype(wT))
    ApplyQuasiArray(*, ChebyshevU{T}(), _BandedMatrix(Fill(-convert(T,π),1,∞), ∞, -1, 1))
end

@simplify function *(H::Hilbert{<:Any,<:ChebyshevInterval}, wU::WeightedBasis{<:Any,<:ChebyshevUWeight,<:ChebyshevU}) 
    T = promote_type(eltype(H), eltype(wU))
    ApplyQuasiArray(*, ChebyshevT{T}(), _BandedMatrix(Fill(convert(T,π),1,∞), ∞, 1, -1))
end


@simplify function *(S::StieltjesPoint, wT::SubQuasiArray{<:Any,2,<:WeightedBasis,<:Tuple{<:AffineQuasiVector,<:Any}})
    P = parent(wT)
    z, x = parent(S).args[1].args
    z̃ = inbounds_getindex(parentindices(wT)[1], z)
    x̃ = axes(P,1)
    (inv.(z̃ .- x̃') * P)[:,parentindices(wT)[2]]
end

@simplify function *(H::Hilbert, wT::SubQuasiArray{<:Any,2,<:WeightedBasis,<:Tuple{<:AffineQuasiVector,<:Any}}) 
    P = parent(wT)
    x = axes(P,1)
    apply(*, inv.(x .- x'), P)[parentindices(wT)...]
end