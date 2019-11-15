##
# Chebyshev
##

struct ChebyshevWeight{T} <: AbstractJacobiWeight{T} end
ChebyshevWeight() = ChebyshevWeight{Float64}()

struct Chebyshev{T} <: AbstractJacobi{T} end
Chebyshev() = Chebyshev{Float64}()
==(a::Chebyshev, b::Chebyshev) = true


function getindex(w::ChebyshevWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    inv(sqrt(1-x^2))
end

struct ChebyshevGrid{kind,T} <: AbstractVector{T}
    n::Int
end

ChebyshevGrid{kind}(n::Integer) where kind = ChebyshevGrid{kind,Float64}(n)

size(g::ChebyshevGrid) = (g.n,)
getindex(g::ChebyshevGrid{1,T}, k::Integer) where T = 
    sinpi(convert(T,g.n-2k+1)/(2g.n))

function getindex(g::ChebyshevGrid{2,T}, k::Integer) where T
    g.n == 1 && return zero(T)
    sinpi(convert(T,g.n-2k+1)/(2g.n-2))
end

function grid(Tn::SubQuasiArray{<:Any,2,<:Chebyshev,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) 
    kr,jr = parentindices(Tn)
    ChebyshevGrid{1,eltype(kr)}(maximum(jr))
end


##
# Ultraspherical
##

struct UltrasphericalWeight{T,Λ} <: AbstractJacobiWeight{T} 
    λ::Λ
end

UltrasphericalWeight(λ) = UltrasphericalWeight{typeof(λ),typeof(λ)}(λ)

function getindex(w::UltrasphericalWeight, x::Number)
    x ∈ axes(w,1) || throw(BoundsError())
    (1-x^2)^(w.λ-one(w.λ)/2)
end



struct Ultraspherical{T,Λ} <: AbstractJacobi{T} 
    λ::Λ
end
Ultraspherical{T}(λ::Λ) where {T,Λ} = Ultraspherical{T,Λ}(λ)
Ultraspherical(λ::Λ) where Λ = Ultraspherical{Float64,Λ}(λ)
Ultraspherical(P::Legendre{T}) where T = Ultraspherical(one(T)/2)
function Ultraspherical(P::Jacobi{T}) where T
    P.a == P.b || throw(ArgumentError("$P is not ultraspherical"))
    Ultraspherical(P.a+one(T)/2)
end

==(a::Ultraspherical, b::Ultraspherical) = a.λ == b.λ

###
# interrelationships
###

Jacobi(C::Ultraspherical{T}) where T = Jacobi(C.λ-one(T)/2,C.λ-one(T)/2)
Jacobi(C::Chebyshev{T}) where T = Jacobi(-one(T)/2,-one(T)/2)




########
# Jacobi Matrix
########

jacobimatrix(C::Chebyshev{T}) where T = 
    _BandedMatrix(Vcat(Fill(one(T)/2,1,∞), 
                        Zeros(1,∞), 
                        Hcat(one(T), Fill(one(T)/2,1,∞))), ∞, 1, 1)

function jacobimatrix(P::Ultraspherical{T}) where T
    λ = P.λ
    _BandedMatrix(Vcat((((2λ-1):∞) ./ (2 .*((zero(T):∞) .+ λ)))',
                        Zeros{T}(1,∞),
                        ((one(T):∞) ./ (2 .*((zero(T):∞) .+ λ)))'), ∞, 1, 1)
end


##########
# Derivatives
##########

# Ultraspherical(1)\(D*Chebyshev())
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::Chebyshev)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix((zero(T):∞)', ∞, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{T}(1), A)
end

# Ultraspherical(1/2)\(D*Legendre())
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::Legendre)
    T = promote_type(eltype(D),eltype(S))
    A = _BandedMatrix(Ones{T}(1,∞), ∞, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{T}(3/2), A)
end


# Ultraspherical(λ+1)\(D*Ultraspherical(λ))
@simplify function *(D::Derivative{<:Any,<:ChebyshevInterval}, S::Ultraspherical)
    A = _BandedMatrix(Fill(2S.λ,1,∞), ∞, -1,1)
    ApplyQuasiMatrix(*, Ultraspherical{eltype(S)}(S.λ+1), A)
end


##########
# Conversion
##########

@simplify \(A::Ultraspherical, B::Legendre) = A\Ultraspherical(B)
@simplify \(A::Legendre, B::Ultraspherical) = Ultraspherical(A)\B

@simplify function \(A::Ultraspherical, B::Jacobi) 
    Ã = Jacobi(A)
    Diagonal(Ã[1,:]./A[1,:]) * (Ã\B)
end
@simplify function \(A::Jacobi, B::Ultraspherical) 
    B̃ = Jacobi(B)
    (A\B̃)*Diagonal(B[1,:]./B̃[1,:])
end

@simplify function \(U::Ultraspherical{<:Any,<:Integer}, C::Chebyshev)
    if U.λ == 1
        T = promote_type(eltype(U), eltype(C))
        _BandedMatrix(Vcat(-Ones{T}(1,∞)/2,
                            Zeros{T}(1,∞), 
                            Hcat(Ones{T}(1,1),Ones{T}(1,∞)/2)), ∞, 0,2)
    elseif U.λ > 0
        (U\Ultraspherical(1)) * (Ultraspherical(1)\C)
    else
        error("Not implemented")
    end
end

@simplify function \(C2::Ultraspherical{<:Any,<:Integer}, C1::Ultraspherical{<:Any,<:Integer})
    λ = C1.λ
    T = promote_type(eltype(C2), eltype(C1))
    if C2.λ == λ+1 
        _BandedMatrix( Vcat(-(λ ./ ((0:∞) .+ λ))', Zeros(1,∞), (λ ./ ((0:∞) .+ λ))'), ∞, 0, 2)
    elseif C2.λ == λ
        Eye{T}(∞)
    elseif C2.λ > λ
        (C2 \ Ultraspherical(λ+1)) * (Ultraspherical(λ+1)\C1)
    else
        error("Not implemented")
    end
end

@simplify function \(C2::Ultraspherical, C1::Ultraspherical)
    λ = C1.λ
    T = promote_type(eltype(C2), eltype(C1))
    if C2.λ == λ+1 
        _BandedMatrix( Vcat(-(λ ./ ((0:∞) .+ λ))', Zeros(1,∞), (λ ./ ((0:∞) .+ λ))'), ∞, 0, 2)
    elseif C2.λ == λ
        Eye{T}(∞)
    else
        error("Not implemented")
    end
end

@simplify function \(w_A::WeightedBasis{<:Any,<:ChebyshevWeight,<:Chebyshev}, w_B::WeightedBasis{<:Any,<:UltrasphericalWeight,<:Ultraspherical}) 
    wA,A = w_A.args
    wB,B = w_B.args
    T = promote_type(eltype(w_A), eltype(w_B))
    @assert wB.λ == B.λ == 1
    _BandedMatrix(Vcat(Fill(one(T)/2, 1, ∞), Zeros{T}(1, ∞), Fill(-one(T)/2, 1, ∞)), ∞, 2, 0)
end

# @simplify function \(w_A::WeightedBasis{<:Any,<:UltrasphericalWeight,<:Ultraspherical}, w_B::WeightedBasis{<:Any,<:UltrasphericalWeight,<:Ultraspherical}) 
#     wA,A = w_A.args
#     wB,B = w_B.args

#     if wA == wB
#         A \ B
#     elseif B.λ == A.λ+1 && wB.λ == wA.λ+1
#         λ = B.λ
#         _BandedMatrix(Vcat((, 
#                             Zeros(1,∞),
#                             ((2:2:∞)./((2:2:∞) .+ (A.a+A.b)))'), ∞, 2,0)

#     elseif wB.a ≥ wA.a+1
#         J = JacobiWeight(wB.b,wB.a-1) .* Jacobi(B.b,B.a-1) 
#         (w_A\J) * (J\w_B)
#     elseif wB.b ≥ wA.b+1
#         J = JacobiWeight(wB.b-1,wB.a) .* Jacobi(B.b-1,B.a) 
#         (w_A\J) * (J\w_B)
#     else
#         error("not implemented for $A and $wB")
#     end
# end


####
# interrelationships
####

# (18.7.3)

@simplify function \(A::Chebyshev, B::Jacobi)
    T = promote_type(eltype(A), eltype(B))
    (B.a == B.b == -T/2) || throw(ArgumentError())
    Diagonal(Jacobi(-T/2,-T/2)[1,:])
end