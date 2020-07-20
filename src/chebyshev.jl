##
# Chebyshev
##

struct ChebyshevWeight{kind,T} <: AbstractJacobiWeight{T} end
ChebyshevWeight{kind}() where kind = ChebyshevWeight{kind,Float64}()
ChebyshevWeight() = ChebyshevWeight{1,Float64}()

struct Chebyshev{kind,T} <: AbstractJacobi{T} end
Chebyshev() = Chebyshev{1,Float64}()
Chebyshev{kind}() where kind = Chebyshev{kind,Float64}()


const WeightedChebyshev{kind,T} = WeightedBasis{T,<:ChebyshevWeight{kind},<:Chebyshev{kind}}

WeightedChebyshev{kind}() where kind = ChebyshevWeight{kind}() .* Chebyshev{kind}()
WeightedChebyshev{kind,T}() where {kind,T} = ChebyshevWeight{kind,T}(λ) .* Chebyshev{kind,T}(λ)

const ChebyshevTWeight = ChebyshevWeight{1}
const ChebyshevUWeight = ChebyshevWeight{2}
const ChebyshevT = Chebyshev{1}
const ChebyshevU = Chebyshev{2}
const WeightedChebyshevT = WeightedChebyshev{1}
const WeightedChebyshevU = WeightedChebyshev{2}

==(a::Chebyshev{kind}, b::Chebyshev{kind}) where kind = true
==(a::Chebyshev, b::Chebyshev) = false
==(::Chebyshev, ::Jacobi) = false
==(::Jacobi, ::Chebyshev) = false


function getindex(w::ChebyshevTWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    inv(sqrt(1-x^2))
end

function getindex(w::ChebyshevUWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    sqrt(1-x^2)
end




Jacobi(C::ChebyshevT{T}) where T = Jacobi(-one(T)/2,-one(T)/2)
Jacobi(C::ChebyshevU{T}) where T = Jacobi(one(T)/2,one(T)/2)


#######
# transform
#######

factorize(L::SubQuasiArray{T,2,<:ChebyshevT,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), plan_chebyshevtransform(Array{T}(undef, size(L,2))))

factorize(L::SubQuasiArray{T,2,<:ChebyshevU,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(L), plan_chebyshevutransform(Array{T}(undef, size(L,2))))


########
# Jacobi Matrix
########

jacobimatrix(C::ChebyshevT{T}) where T =
    _BandedMatrix(Vcat(Fill(one(T)/2,1,∞),
                        Zeros{T}(1,∞),
                        Hcat(one(T), Fill(one(T)/2,1,∞))), ∞, 1, 1)

jacobimatrix(C::ChebyshevU{T}) where T =
    _BandedMatrix(Vcat(Fill(one(T)/2,1,∞),
                        Zeros{T}(1,∞),
                        Fill(one(T)/2,1,∞)), ∞, 1, 1)



# These return vectors A[k], B[k], C[k] are from DLMF. 
recurrencecoefficients(C::ChebyshevT) = (Vcat(1, Fill(2,∞)), Zeros{Int}(∞), Ones{Int}(∞))
recurrencecoefficients(C::ChebyshevU) = (Fill(2,∞), Zeros{Int}(∞), Ones{Int}(∞))

# special clenshaw!
function copyto!(dest::AbstractVector{T}, v::SubArray{<:Any,1,<:Expansion{<:Any,<:ChebyshevT}, <:Tuple{AbstractVector{<:Number}}}) where T
    f = parent(v)
    (x,) = parentindices(v)
    P,c = arguments(f)
    clenshaw!(paddeddata(c), x, dest)
end

##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::ChebyshevT)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix((zero(T):∞)', ∞, -1,1)
    ApplyQuasiMatrix(*, ChebyshevU{T}(), A)
end

#####
# Conversion
#####

@simplify function \(U::ChebyshevU, C::ChebyshevT)
    T = promote_type(eltype(U), eltype(C))
    _BandedMatrix(Vcat(-Ones{T}(1,∞)/2,
                        Zeros{T}(1,∞),
                        Hcat(Ones{T}(1,1),Ones{T}(1,∞)/2)), ∞, 0,2)
end

@simplify function \(w_A::WeightedChebyshevT, w_B::WeightedChebyshevU)
    wA,A = w_A.args
    wB,B = w_B.args
    T = promote_type(eltype(w_A), eltype(w_B))
    _BandedMatrix(Vcat(Fill(one(T)/2, 1, ∞), Zeros{T}(1, ∞), Fill(-one(T)/2, 1, ∞)), ∞, 2, 0)
end


####
# interrelationships
####

# (18.7.3)

@simplify function \(A::ChebyshevT, B::Jacobi)
    J = Jacobi(A)
    Diagonal(J[1,:]) * (J \ B)
end

@simplify function \(A::Jacobi, B::ChebyshevT)
    J = Jacobi(B)
    (A \ J) * Diagonal(inv.(J[1,:]))
end

@simplify function \(A::Chebyshev, B::Jacobi)
    J = Jacobi(A)
    Diagonal(A[1,:] .\ J[1,:]) * (J \ B)
end

@simplify function \(A::Jacobi, B::Chebyshev)
    J = Jacobi(B)
    (A \ J) * Diagonal(J[1,:] .\ B[1,:])
end

@simplify function \(A::Jacobi, B::ChebyshevU)
    T = promote_type(eltype(A), eltype(B))
    (A.a == A.b == one(T)/2) || throw(ArgumentError())
    Diagonal(B[1,:] ./ A[1,:])
end

# TODO: Toeplitz dot Hankel will be faster to generate
@simplify function \(A::ChebyshevT, B::Legendre)
    T = promote_type(eltype(A), eltype(B))
   UpperTriangular( BroadcastMatrix{T}((k,j) -> begin
            (iseven(k) == iseven(j) && j ≥ k) || return zero(T)
            k == 1 && return Λ(convert(T,j-1)/2)^2/π
            2/π * Λ(convert(T,j-k)/2) * Λ(convert(T,k+j-2)/2)
        end, 1:∞, (1:∞)'))
end


@simplify function \(A::Jacobi, B::WeightedBasis{<:Any,<:JacobiWeight,<:Chebyshev})
    w, T = B.args
    J = Jacobi(T)
    wJ = w .* J
    (A \ wJ) * (J \ T)
end

@simplify function \(A::Chebyshev, B::WeightedBasis{<:Any,<:JacobiWeight,<:Jacobi})
    J = Jacobi(A)
    (A \ J) * (J \ B)
end

@simplify function \(A::Chebyshev, B::WeightedBasis{<:Any,<:JacobiWeight,<:Chebyshev})
    J = Jacobi(A)
    (A \ J) * (J \ B)
end



####
# sum
####

function _sum(A::WeightedBasis{T,<:ChebyshevUWeight,<:ChebyshevU}, dims) where T
    w, U = A.args
    @assert dims == 1
    Hcat(convert(T, π)/2, Zeros{T}(1,∞))
end

function _sum(A::WeightedBasis{T,<:ChebyshevWeight,<:Chebyshev}, dims) where T
    @assert dims == 1
    Hcat(convert(T, π), Zeros{T}(1,∞))
end


####
# algebra
####

broadcastbasis(::typeof(+), ::ChebyshevT, U::ChebyshevU) = U
broadcastbasis(::typeof(+), U::ChebyshevU, ::ChebyshevT) = U